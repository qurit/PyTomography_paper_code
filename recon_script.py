
import sys
sys.path.append('/home/gpuvmadm/PyTomography/src')
import numpy as np
import os
from pytomography.transforms import SPECTAttenuationTransform, SPECTPSFTransform, KEMTransform
from pytomography.projectors import SPECTSystemMatrix
from pytomography.io.SPECT import simind
from pytomography.algorithms import BSREM, OSEM, KEM
from pytomography.priors import RelativeDifferencePrior, TopNAnatomyNeighbourWeight, AnatomyNeighbourWeight
from misc import get_organ_masks, get_organ_volume, get_photopeak_scatter, get_activities_pct, get_psf_meta, SaveData
import torch
import time

def reconstruct_phantom(organ_specifications_path, organ_concentrations_path, organ_concentrations_index, scatter_type, projection_time, CPSperMBq, n_iters, save_path, algorithm, algorithm_kwargs, masks=None, mask_volumes=None, save_recon_object=False, activities_true=None, random_seed=None):
    
    if random_seed is not None:
        torch.manual_seed(random_seed)
        
    if projection_time==np.inf:
        dT = 10
    else:
        dT = projection_time
    
    # Get organ labels
    _, labels, _, _, _ = np.genfromtxt(organ_specifications_path, delimiter=',', skip_header=3, dtype=str).T
    
    # Get organ volumes
    if mask_volumes is None:
        _, _, GT_paths, _, _ = np.genfromtxt(organ_specifications_path, delimiter=',', skip_header=3, dtype=str).T
        GT_dz, GT_dy, GT_dx = np.genfromtxt(organ_specifications_path, delimiter=',', skip_header=1, max_rows=1).T
        mask_volumes = np.vectorize(get_organ_volume)(GT_paths, GT_dx*GT_dy*GT_dz)
        
    # Open MetaData
    if activities_true is None:
        activities_true =  np.genfromtxt(organ_concentrations_path, delimiter=',').T[organ_concentrations_index+1]
    object_meta, proj_meta, photopeak, scatter_TEW = get_photopeak_scatter(organ_specifications_path, organ_concentrations_path, organ_concentrations_index, dT, headerfile_peak='photopeak.h00', headerfile_lower='lowerscatter.h00', headerfile_upper='upperscatter.h00')
    _, _, primary, _ = get_photopeak_scatter(organ_specifications_path, organ_concentrations_path, organ_concentrations_index, dT, headerfile_peak='primary.h00', headerfile_lower='lowerscatter.h00', headerfile_upper='upperscatter.h00')
    
    if scatter_type=='TEW':
        scatter = scatter_TEW
    elif scatter_type=='true':
        scatter = photopeak - primary
    
    if projection_time is not np.inf:
        photopeak = torch.poisson(photopeak)
        scatter = torch.poisson(scatter)

    calibration_factor = 1 / (dT * CPSperMBq * object_meta.dr[0]**3)
    
    # Option to give masks as input argument to speed things up and not have to run this
    if masks is None:
        masks = get_organ_masks(organ_specifications_path, object_meta, full_voxel=False)
    
    attenuation_map = simind.get_attenuation_map(os.path.join('/disk1/EANM2023/mu208.hct'))[:,:,128:256]
    att_transform = SPECTAttenuationTransform(attenuation_map)
    psf_meta = get_psf_meta(organ_specifications_path, 'photopeak.h00')
    psf_transform = SPECTPSFTransform(psf_meta)
    
    if algorithm==BSREM:
        prior_weight = TopNAnatomyNeighbourWeight(attenuation_map, 8)
        prior = RelativeDifferencePrior(beta=0.3, gamma=2, weight=prior_weight)
        algorithm_kwargs['prior'] = prior
        
    if algorithm==KEM:
        algorithm_kwargs['kem_transform'] = KEMTransform(
        support_objects=[attenuation_map],
        support_kernels_params=[[0.005]],
        distance_kernel_params=[0.5],
        kernel_on_gpu=True,
        top_N = 40)
        
    system_matrix = SPECTSystemMatrix(
        obj2obj_transforms = [att_transform,psf_transform],
        proj2proj_transforms = [],
        object_meta = object_meta,
        proj_meta = proj_meta)
    
    reconstruction_algorithm = algorithm(
        projections = photopeak,
        system_matrix = system_matrix,
        scatter=scatter,
        **algorithm_kwargs
        )
    
    callback  = SaveData(calibration_factor, n_subset_save=7, masks=masks)
    start = time.time()
    reconstructed_object = reconstruction_algorithm(n_iters=n_iters, n_subsets=8, callback=callback)
    time_elapsed = time.time() - start
    
    # Save all data for easy reading
    activity_concs_true = activities_true
    activity_concs_predicted = np.array(callback.activities).T
    activity_concs_predicted_noise = np.array(callback.activities_noise).T
    activity_concs_pct = get_activities_pct(activity_concs_predicted, activities_true)
    activity_concs_pct_noise = get_activities_pct(activity_concs_predicted_noise, activities_true)
    activity_totals_true = activities_true * mask_volumes
    activity_totals_predicted = (np.array(callback.activities)* mask_volumes).T
    np.savez(save_path,
             activity_concs_true=activity_concs_true, activity_concs_predicted=activity_concs_predicted, activity_concs_pct=activity_concs_pct,
             activity_concs_predicted_noise = activity_concs_predicted_noise,
             activity_concs_pct_noise = activity_concs_pct_noise,
             activity_totals_true=activity_totals_true,activity_totals_predicted=activity_totals_predicted, mask_volumes=mask_volumes, labels=labels,
             time_elapsed = time_elapsed)
    if save_recon_object:
        reconstructed_object = reconstructed_object[0].cpu().numpy() * calibration_factor
        np.save(file=f'{save_path}_object', arr=reconstructed_object)
    