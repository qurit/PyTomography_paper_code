import torch
import pytomography
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta, PETTOFMeta
from pytomography.projectors.PET import PETLMSystemMatrix
from pytomography.algorithms import OSEM
from pytomography.io.PET import gate
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.transforms.shared import GaussianFilter
from pytomography.utils import sss
import gc
import os
import numpy as np
import misc
import time
import pickle
import itk

input_path = '/disk1/pytomography_paper_data/input'
output_path = '/disk1/pytomography_paper_data/output'

# GT images
pet_aligned = misc.align_highres_image(os.path.join(input_path, 'pet_gate_experiment', 'fdg_pet_phantom_uptake.nii.gz'))
pet_masks = np.transpose(itk.GetArrayFromImage(itk.imread(os.path.join(input_path, 'pet_gate_experiment', 'pet_masks.seg.nrrd'))), (1,2,0))
mask_whitematter = pet_masks==1
mask_greymatter = pet_masks==2

# Info
info = gate.get_detector_info(path = os.path.join(input_path, 'pet_gate_experiment', 'mMR_Geometry.mac'), mean_interaction_depth=9)

# TOF
speed_of_light = 0.3 #mm/ps
fwhm_tof_resolution = 550 * speed_of_light / 2 #ps to position along LOR
TOF_range = 1000 * speed_of_light #ps to position along LOR (full range)
num_tof_bins = 5
tof_meta = PETTOFMeta(num_tof_bins, TOF_range, fwhm_tof_resolution, n_sigmas=3)

# Detector IDs
detector_ids = torch.load(os.path.join(input_path, 'pet_gate_experiment', 'detector_ids', 'detector_ids_tof_all_events.pt'))
num_events = detector_ids.shape[0]
detector_ids = detector_ids[:int(num_events/3)] # only 1/3 events
norm_factor = torch.load(os.path.join(input_path, 'pet_gate_experiment', 'detector_ids', 'norm_factor_lm.pt'))

# Meta
object_meta = ObjectMeta(
    dr=(2,2,2), #mm
    shape=(128,128,96) #voxels
)
proj_meta = PETLMProjMeta(
    detector_ids,
    info,
    tof_meta=tof_meta,
    weights_sensitivity=norm_factor
    )

start = time.time()
atten_map = gate.get_attenuation_map_nifti(os.path.join(input_path, 'pet_gate_experiment', 'fdg_pet_phantom_umap.nii.gz'), object_meta)
psf_transform = GaussianFilter(4)
system_matrix = PETLMSystemMatrix(
       object_meta,
       proj_meta,
       obj2obj_transforms = [psf_transform],
       N_splits=10,
       attenuation_map=atten_map.to(pytomography.device),
)
end = time.time()
print(f'Time to compute system matrix: {end-start}')

# ----------------------------------
# RANDOM PLUS SCATTER ESTIMATION
# ---------------------------------

# Randoms
start = time.time()
detector_ids_randoms= torch.load(os.path.join(input_path, 'pet_gate_experiment', 'detector_ids', 'detector_ids_delays.pt'))
# Only 1/3rd of events for 3min scan
num_events = detector_ids_randoms.shape[0]
detector_ids_randoms = detector_ids_randoms[:int(num_events/3)] 

sinogram_random = gate.listmode_to_sinogram(detector_ids_randoms, info)
sinogram_random = gate.smooth_randoms_sinogram(sinogram_random, info, sigma_r=4, sigma_theta=4, sigma_z=4)
sinogram_random = gate.randoms_sinogram_to_sinogramTOF(sinogram_random, tof_meta, coincidence_timing_width = 4300) # we need to keep this for scatter estimation later
lm_randoms = gate.sinogram_to_listmode(detector_ids, sinogram_random, info)
end = time.time()
print(f'Time for Random Estimation: {end-start}')

# Scatters
start = time.time()
# Get additive term (without scatter term):
lm_norm = system_matrix._compute_sensitivity_projection(all_ids=False)
additive_term = lm_randoms / lm_norm
additive_term[additive_term.isnan()] = 0 # remove NaN values
# Recon
likelihood = PoissonLogLikelihood(
        system_matrix,
        additive_term = additive_term
    )
recon_algorithm = OSEM(likelihood)
recon_without_scatter_estimation = recon_algorithm(40,1)
scatter_sinogram = sss.get_sss_scatter_estimate(
        object_meta,
        proj_meta,
        recon_without_scatter_estimation,
        atten_map,
        system_matrix,
        tof_meta=tof_meta,
        sinogram_random=sinogram_random)
lm_scatter = gate.sinogram_to_listmode(proj_meta.detector_ids, scatter_sinogram, proj_meta.info)
# Save memory
del(scatter_sinogram)
del(sinogram_random)
gc.collect()
end = time.time()
print(f'Time for TOF-SSS Scatter Estimation: {end-start}')
# ----------------------------------
# END RANDOM PLUS SCATTER ESTIMATION
# ---------------------------------

additive_term = (lm_scatter + lm_randoms) / lm_norm
additive_term[additive_term.isnan()] = 0
likelihood = PoissonLogLikelihood(
        system_matrix,
        additive_term = additive_term
)
cb = misc.StatisticsCallback(pet_aligned, mask_greymatter, mask_whitematter)
recon_algorithm = OSEM(likelihood)

start = time.time()
recon = recon_algorithm(n_iters=80, n_subsets=1, callback=cb)
end = time.time()
print(f'Time to reconstruct: {end-start}')

np.save(os.path.join(output_path, 'pet_gate_experiment', 'pet_recon2.npy'), recon.cpu().numpy())
with open(os.path.join(output_path, 'pet_gate_experiment', 'pet_recon2_callback'), 'wb') as f:
    pickle.dump(cb, f)
