import sys
sys.path.append('/home/gpuvmadm/PyTomography/src')
import numpy as np
import os
from skimage.transform import resize
from monai.transforms import Resize
import pytomography
from pytomography.io.SPECT import simind
from pytomography.callbacks import CallBack
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytomography.device = device

def get_organ_volume(path, GT_dV):
    arr = np.fromfile(path, dtype=np.float32)
    arr[arr>0] = 1
    return np.sum(arr)*GT_dV

def get_psf_meta(organ_specifications_path, headerfile):
    _, _, _, projection_folders, _ = np.genfromtxt(organ_specifications_path, delimiter=',', skip_header=3, dtype=str).T
    headerfile_path = os.path.join(projection_folders[0], headerfile)
    return simind.get_psfmeta_from_header(headerfile_path)

def get_photopeak_scatter(organ_specifications_path, organ_concentrations_path, organ_concentrations_index, dT, headerfile_peak, headerfile_lower, headerfile_upper):
    
    # Get required info
    _, _, GT_paths, projection_folders, regions = np.genfromtxt(organ_specifications_path, delimiter=',', skip_header=3, dtype=str).T
    GT_dz, GT_dy, GT_dx = np.genfromtxt(organ_specifications_path, delimiter=',', skip_header=1, max_rows=1).T
    organ_volumes = np.vectorize(get_organ_volume)(GT_paths, GT_dx*GT_dy*GT_dz)
    concentrations =  np.genfromtxt(organ_concentrations_path, delimiter=',').T[1+organ_concentrations_index]
    # Get projections
    photopeak = 0
    scatter = 0
    for projection_folder, concentration, organ_volume in zip(projection_folders, concentrations, organ_volumes):
        headerfile_peak_path = os.path.join(projection_folder, headerfile_peak)
        headerfile_lower_path = os.path.join(projection_folder, headerfile_lower)
        headerfile_upper_path = os.path.join(projection_folder, headerfile_upper)
        object_meta, proj_meta = simind.get_metadata(headerfile_peak_path)
        photopeak_i = simind.get_projections(headerfile_peak_path)
        scatter_i = simind.get_scatter_from_TEW(headerfile_peak_path, headerfile_lower_path, headerfile_upper_path)
        photopeak += photopeak_i * concentration * organ_volume * dT # (counts/s/MBq)*(MBq/mL)*mL*s = counts
        scatter += scatter_i * concentration * organ_volume *  dT
    return object_meta, proj_meta, photopeak, scatter

def get_organ_masks(organ_specifications_path, object_meta, GT_dtype=np.float32, scale=True, full_voxel=False, index=None):
    # Get required info
    _, _, GT_paths, _, regions = np.genfromtxt(organ_specifications_path, delimiter=',', skip_header=3, dtype=str).T
    regions = regions.astype(int)
    GT_ordering = np.genfromtxt(organ_specifications_path, delimiter=',', skip_header=2, max_rows=1).T.astype(int)
    GT_shape = np.genfromtxt(organ_specifications_path, delimiter=',', max_rows=1).T.astype(int)
    GT_dz, GT_dy, GT_dx = np.genfromtxt(organ_specifications_path, delimiter=',', skip_header=1, max_rows=1).T
    organ_volumes = np.vectorize(get_organ_volume)(GT_paths, GT_dx*GT_dy*GT_dz)
    # Get masks
    masks = []
    for i, GT_path in enumerate(GT_paths):
        if index is not None:
            if index!=i:
                continue
        GTi = np.fromfile(GT_path, dtype=GT_dtype)
        GTi = GTi.reshape(GT_shape)
        GTi = np.transpose(GTi, GT_ordering)
        GTi = (GTi>0).astype(np.float32)
        if scale:
            GTi = resize(GTi, object_meta.shape, anti_aliasing=True)
            if full_voxel:
                GTi = (GTi>=1).astype(np.float32)
        masks.append(GTi)
    # TODO write code to adjust for overlapping masks
    # Adjust masks if not scaled
    #if scale:
    #    dV = np.prod(object_meta.dr)
    #    for i in range(len(masks)):
    #        masks[i] = masks[i].astype(np.float32)
    #        masks[i] = masks[i]*organ_volumes[i]/(masks[i].sum()*dV)
    return torch.tensor(np.array(masks)).to(device)

def get_activities_pct(activities, concentrations):
    activities_pct = activities/concentrations[:,np.newaxis] * 100
    return activities_pct

class SaveDataDiscrete(CallBack):
    def __init__(self, calibration_factor, device = None, n_subset_save=0):
        self.device = pytomography.device if device is None else device
        self.calibration_factor = calibration_factor
        self.activities = []
        self.n_subset_save = n_subset_save
    def run(self, activities, n_iter):
        activities_cal = activities[0] *self.calibration_factor
        self.activities.append(activities_cal.cpu().numpy())
            
class SaveData(CallBack):
    def __init__(self, calibration_factor, masks, device = None, n_subset_save=0):
        self.device = pytomography.device if device is None else device
        self.masks = masks.to(self.device)
        self.masks_organonly = self.masks>=1
        self.norm_constant = self.masks.sum(axis=(-1,-2,-3))
        self.norm_constant_organonly = self.masks_organonly.sum(axis=(-1,-2,-3))
        self.calibration_factor = calibration_factor
        self.activities = []
        self.activities_noise = []
        self.activities_organonly = []
        self.activities_noise_organonly = []
        self.n_subset_save = n_subset_save  
        
    def append_activities(self, object, masks, norm_constant, activities_list, activities_noise_list):
        activities = (object.unsqueeze(dim=1) * masks).sum(axis=(-1,-2,-3)) / norm_constant
        activities_cal = activities[0] * self.calibration_factor
        # Compute variances
        activity_noise = []
        for mask in masks:
            activity_noise.append(torch.std(object[mask.unsqueeze(dim=0)>0]).cpu())
        activities_noise_list.append(np.array(activity_noise))
        activities_list.append(activities_cal.cpu().numpy())
        
    def run(self, object, n_iter):
        self.append_activities(object, self.masks, self.norm_constant, self.activities, self.activities_noise)
        # Only for voxels 100% in the organ
        self.append_activities(object, self.masks_organonly, self.norm_constant_organonly, self.activities_organonly, self.activities_noise_organonly)
            
class SaveDataMaskCutoff(CallBack):
    def __init__(self, calibration_factor, masks, device = None, n_subset_save=0):
        self.device = pytomography.device if device is None else device
        self.masks = masks.to(self.device)
        self.masks = (self.masks > 0.9) * self.masks
        self.norm_constant = self.masks.sum(axis=(-1,-2,-3))
        self.calibration_factor = calibration_factor
        self.activities = []
        self.n_subset_save = n_subset_save
        
    def run(self, object, n_iter):
        activities = (object.unsqueeze(dim=1) * self.masks).sum(axis=(-1,-2,-3))/ self.norm_constant
        activities_cal = activities[0] *self.calibration_factor
        self.activities.append(activities_cal.cpu().numpy())
            
class SaveDataScaleUp(CallBack):
    def __init__(self, calibration_factor, masks, device = None, n_subset_save=0):
        self.device = pytomography.device if device is None else device
        self.masks = masks.to(self.device)
        self.norm_constant = masks.sum(axis=(-1,-2,-3))
        self.calibration_factor = calibration_factor
        self.activities = []
        self.n_subset_save = n_subset_save
        self.resizer = Resize((512,512,768))
    def run(self, object, n_iter):
        object = self.resizer(object)
        activities = torch.tensor([(object * mask).sum(axis=(-1,-2,-3))/NC for mask, NC in zip(self.masks, self.norm_constant)])
        # Prevents GPU OOM
        #activities = (object.unsqueeze(dim=1) * self.masks).sum(axis=(-1,-2,-3))/ self.norm_constant
        activities_cal = activities * self.calibration_factor
        self.activities.append(activities_cal.numpy())
        