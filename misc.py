from __future__ import annotations
import torch
import torch.nn as nn
import sys
sys.path.append('/home/gpuvmadm/PyTomography/src')
import numpy as np
from pytomography.callbacks import Callback
import torch
import numpy as np
import torch
import pytomography
import numpy.linalg as npl
from scipy.ndimage import affine_transform, binary_erosion
from torch.optim import LBFGS
import nibabel as nib
from pytomography.callbacks import Callback
from monai.transforms import ScaleIntensityd, CropForeground, Compose, DivisiblePadd, SpatialCropd, ThresholdIntensityd

def align_highres_image(path, img=None, dr=(2,2,2), shape=(128,128,96)):
    # If img is none, extract data from path
    data = nib.load(path)
    # If img is none, extract data from path
    if img is None:
        img = data.get_fdata()
    Sx, Sy, Sz = -(np.array(img.shape)-1) / 2
    dx, dy, dz = data.header['pixdim'][1:4]
    # Convert from RAS to LPS space for DICOM
    dx*=-1; dy*=-1
    M_highres = np.zeros((4,4))
    M_highres[0] = np.array([dx, 0, 0, Sx*dx])
    M_highres[1] = np.array([0, dy, 0, Sy*dy])
    M_highres[2] = np.array([0, 0, dz, Sz*dz])
    M_highres[3] = np.array([0, 0, 0, 1])
    dx, dy, dz = dr
    Sx, Sy, Sz = -(np.array(shape)-1) / 2
    M_pet = np.zeros((4,4))
    M_pet[0] = np.array([dx, 0, 0, Sx*dx])
    M_pet[1] = np.array([0, dy, 0, Sy*dy])
    M_pet[2] = np.array([0, 0, dz, Sz*dz])
    M_pet[3] = np.array([0, 0, 0, 1])
    M = npl.inv(M_highres) @ M_pet
    return affine_transform(img, M, output_shape=shape, mode='constant', order=1)

def get_masks(pet_path, grey_matter_upper=np.inf, grey_matter_lower=0.6, white_matter_upper=0.4, white_matter_lower=0.15, alignment_overlap=0.8):
    pet_data = nib.load(pet_path)
    pet_highres = pet_data.get_fdata()
    mask_greymatter = (pet_highres<grey_matter_upper)*(pet_highres>grey_matter_lower)
    mask_whitematter = (pet_highres<white_matter_upper)*(pet_highres>white_matter_lower)
    mask_whitematter = binary_erosion(mask_whitematter, iterations=1)
    mask_greymatter = binary_erosion(mask_greymatter, iterations=1)
    grey_matter_mask_aligned = align_highres_image(pet_path, mask_greymatter.astype(float))>alignment_overlap
    white_matter_mask_aligned = align_highres_image(pet_path, mask_whitematter.astype(float))>alignment_overlap
    return grey_matter_mask_aligned, white_matter_mask_aligned

def compute_CRC(img, pet_aligned, mask_greymatter, mask_whitematter):
    ar = img[mask_greymatter].mean()
    br = img[mask_whitematter].mean()
    atrue = pet_aligned[mask_greymatter].mean()
    btrue = pet_aligned[mask_whitematter].mean()
    return (ar/br-1)/(atrue/btrue-1)

def compute_mse(img, pet_aligned, mask):
    ratio = pet_aligned.sum() / img.sum()
    true_mean = pet_aligned[mask].mean()
    bias = (ratio*img - pet_aligned)[mask].mean()
    std = (ratio*img - pet_aligned)[mask].std()
    return bias/true_mean, std/true_mean

class StatisticsCallback(Callback):
    def __init__(
        self,
        pet_aligned,
        mask_greymatter,
        mask_whitematter
    ) -> None:
        self.pet_aligned = pet_aligned
        self.mask_greymatter = mask_greymatter
        self.mask_whitematter = mask_whitematter
        self.CRCs = []
        self.biass = []
        self.stds = []
        self.biass_wm = []
        self.stds_wm = []
        self.biass_gm = []
        self.stds_gm = []
    def run(self, object: torch.tensor, n_iter: int, n_subset: int):
        CRC = compute_CRC(object.cpu().numpy(), self.pet_aligned, self.mask_greymatter, self.mask_whitematter)
        bias, std = compute_mse(object.cpu().numpy(), self.pet_aligned, None)
        bias_wm, std_wm = compute_mse(object.cpu().numpy(), self.pet_aligned, self.mask_whitematter)
        bias_gm, std_gm = compute_mse(object.cpu().numpy(), self.pet_aligned, self.mask_greymatter)
        print(f'Bias WM: {bias_wm}')
        print(f'std WM: {std_wm}')
        print(f'Bias GM: {bias_gm}')
        print(f'std GM: {std_gm}')
        self.CRCs.append(CRC)
        self.biass.append(bias)
        self.stds.append(std)
        self.biass_wm.append(bias_wm)
        self.stds_wm.append(std_wm)
        self.biass_gm.append(bias_gm)
        self.stds_gm.append(std_gm)
        return object
        
def get_pipeline(mri_aligned, mri_crop_above, mri_crop_below):
    roi_start, roi_end = CropForeground().compute_bounding_box(mri_aligned)
    pipeline = Compose([
        SpatialCropd(['MR', 'NM'], roi_start=roi_start, roi_end=roi_end, allow_missing_keys=True),
        DivisiblePadd(['MR', 'NM'], 16, allow_missing_keys=True),
        ThresholdIntensityd(['MR'], mri_crop_above, above=False, cval=mri_crop_above),
        ThresholdIntensityd(['MR'], mri_crop_below, above=True, cval=mri_crop_below),
        ScaleIntensityd(['MR'], 0, 1)
    ])
    return pipeline
class DIPPrior():
    def __init__(
        self,
        network,
        anatomical_image,
        pipeline, # pipeline for preprocessing MRI image
        scale_factor=1, # constant to scale MRI image by
        n_epochs=10, # how many epochs the network trains for when fitting
        lr = 0.1, # learning rate when fitting
    ):
        self.network = network
        self.anatomical_image = anatomical_image
        self.pipeline = pipeline
        self.n_epochs = n_epochs
        self.scale_factor = scale_factor
        self.lr = lr
        self.max_iter = 20

    def fit(self, object):
        # This method trains the network for n_epochs at a learning rate of lr
        data = self.pipeline({'NM': object, 'MR': self.anatomical_image})
        optimizer_lfbgs = LBFGS(self.network.parameters(), lr=self.lr, max_iter=self.max_iter, history_size=100)
        NM_truth = data['NM'].unsqueeze(0).unsqueeze(0) * self.scale_factor
        network_input = data['MR'].unsqueeze(0).unsqueeze(0)
        criterion = torch.nn.MSELoss()
        def closure(optimizer):
            optimizer.zero_grad()
            NM_prediction = self.network(network_input)
            loss = criterion(NM_prediction, NM_truth)
            loss.backward()
            return loss
        for epoch in range(self.n_epochs):
            loss = optimizer_lfbgs.step(lambda: closure(optimizer_lfbgs))
        self.network.zero_grad(set_to_none=True)
        with torch.no_grad():
            # Add batch/channel dimension
            network_prediction = self.network(data['MR'].unsqueeze(0).unsqueeze(0)).squeeze()
        self.prior_object = self.pipeline.inverse({'NM': network_prediction})['NM'].as_tensor() / self.scale_factor

    def predict(self):
        return self.prior_object.detach()

def get_downward_block(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(), 
        ) 
    
def get_downsample_block(out_channels):
    return nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=(1,1,1)),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
        ) 
    
def get_bottleneck_block(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
        ) 
    
def get_bilinear_upsample_block(in_channels, out_channels):
    return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, padding='same'),
        )
    
def get_upward_block(in_channels):
    return nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding='same'),
            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding='same'),
            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(),
        )
    
def get_final_block(in_channels):
    return nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding='same'),
            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels, 1, kernel_size=3, padding='same'),
        )
    
class UNetCustom(nn.Module):
    def __init__(self, n_channels=[4, 8, 16, 32, 64]):
        super().__init__()       
        self.downward_block1 = get_downward_block(1, n_channels[0])
        self.downward_block2 = get_downward_block(n_channels[0], n_channels[1])
        self.downward_block3 = get_downward_block(n_channels[1], n_channels[2])
        self.downward_block4 = get_downward_block(n_channels[2], n_channels[3])
        self.downsample_block1 = get_downsample_block(n_channels[0])
        self.downsample_block2 = get_downsample_block(n_channels[1])
        self.downsample_block3 = get_downsample_block(n_channels[2])
        self.downsample_block4 = get_downsample_block(n_channels[3])
        self.bottleneck_block = get_bottleneck_block(n_channels[3], n_channels[4])
        self.upsample_block1 = get_bilinear_upsample_block(n_channels[4], n_channels[3])
        self.upsample_block2 = get_bilinear_upsample_block(n_channels[3], n_channels[2])
        self.upsample_block3 = get_bilinear_upsample_block(n_channels[2], n_channels[1])
        self.upsample_block4 = get_bilinear_upsample_block(n_channels[1], n_channels[0])
        self.upward_block1 = get_upward_block(n_channels[3])
        self.upward_block2 = get_upward_block(n_channels[2])
        self.upward_block3 = get_upward_block(n_channels[1])
        self.final_block = get_final_block(n_channels[0])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.downward_block1(x)
        x = self.downsample_block1(x1)
        x2 = self.downward_block2(x)
        x = self.downsample_block2(x2)
        x3 = self.downward_block3(x)
        x = self.downsample_block3(x3)
        x4 = self.downward_block4(x)
        x = self.downsample_block4(x4)
        x = self.bottleneck_block(x)
        x = self.upsample_block1(x) + x4
        x = self.upward_block1(x)
        x = self.upsample_block2(x) + x3
        x = self.upward_block2(x)
        x = self.upsample_block3(x) + x2
        x = self.upward_block3(x)
        x = self.upsample_block4(x) + x1
        x = self.final_block(x)
        return x


        
        
        