
import sys
sys.path.append('/home/gpuvmadm/PyTomography/src')
import numpy as np
import pytomography
from misc import get_organ_masks, get_organ_volume, get_photopeak_scatter
import torch
import recon_script
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytomography.device = device

organ_specifications_path = '/disk1/pytomography_paper_results/data/organ_specifications.csv'
organ_concentrations_path = '/disk1/pytomography_paper_results/data/organ_concentrations.csv'
organ_concentrations_indices = [0,1,2,3,4,5,6,7,8,9]
recon_types = ['regular']
scatter_types = ['TEW']
organ_segmentation_type = 'GT'
projection_times = [15]
prior_types = [0,1,2]
prior_names = ['noprior', 'RDP', 'RDPAP']
CPSperMBq = 11.7284

object_meta, _, _, _ = get_photopeak_scatter(organ_specifications_path, organ_concentrations_path, 0, dT=1, headerfile_peak='photopeak.h00', headerfile_lower='lowerscatter.h00', headerfile_upper='upperscatter.h00')
masks = get_organ_masks(organ_specifications_path, object_meta, full_voxel=True)
_, _, GT_paths, _, _ = np.genfromtxt(organ_specifications_path, delimiter=',', skip_header=3, dtype=str).T
GT_dz, GT_dy, GT_dx = np.genfromtxt(organ_specifications_path, delimiter=',', skip_header=1, max_rows=1).T
mask_volumes = np.vectorize(get_organ_volume)(GT_paths, GT_dx*GT_dy*GT_dz)


for organ_concentrations_index in organ_concentrations_indices:
    for recon_type in recon_types:
        for scatter_type in scatter_types:
            for projection_time in projection_times:
                for prior_type, prior_name in zip(prior_types, prior_names):
                    if recon_type=='masked':
                        n_iters = 15
                    elif recon_type=='regular':
                        n_iters = 120
                    print('next')
                    save_path = f'/disk1/multibody_new/{organ_concentrations_path.split("/")[-1][:-4]}_{organ_concentrations_index}_recon{recon_type}_{prior_name}_scat{scatter_type}_dT{projection_time}_masks{organ_segmentation_type}_niters{n_iters}'
                    recon_script.reconstruct_phantom(organ_specifications_path, organ_concentrations_path, organ_concentrations_index, recon_type, scatter_type, organ_segmentation_type, projection_time, CPSperMBq, n_iters, masks=masks, save_path=save_path, prior_type=prior_type, save_recon_object=True)