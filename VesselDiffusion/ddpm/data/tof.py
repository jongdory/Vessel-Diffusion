import os
import numpy as np
import pandas as pd
import albumentations
from torch.utils.data import Dataset

from monai import transforms
from monai.data import Dataset as MonaiDataset

TOF_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "frangi", "mask"]),
        transforms.EnsureChannelFirstd(keys=["image", "frangi", "mask"]),
        transforms.CropForegroundd(keys=["image", "frangi", "mask"], source_key="mask",),
        transforms.SpatialPadd(keys=["image", "frangi", "mask"], spatial_size=(160, 160, 160)),
        transforms.Lambdad(keys=["image", "frangi"], func=lambda x: x[0, :, :, :]),
        transforms.AddChanneld(keys=["image", "frangi"]),
        transforms.EnsureTyped(keys=["image", "frangi"]),
        transforms.Orientationd(keys=["image", "frangi", "mask"], axcodes="RPI"),
        transforms.RandSpatialCropd(keys=["image", "frangi", "mask"], roi_size=(128, 128, 128), random_size=False),
        transforms.ScaleIntensityRangePercentilesd(keys=["image", "frangi"], lower=0, upper=100, b_min=0, b_max=1),
    ]
)

TOF_testforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "frangi", "mask"]),
        transforms.EnsureChannelFirstd(keys=["image", "frangi", "mask"]),
        transforms.Lambdad(keys=["image", "frangi"], func=lambda x: x[0, :, :, :]),
        transforms.AddChanneld(keys=["image", "frangi"]),
        transforms.EnsureTyped(keys=["image", "frangi"]),
        transforms.Orientationd(keys=["image", "frangi", "mask"], axcodes="RPI"),
        transforms.RandSpatialCropd(keys=["image", "frangi", "mask"], roi_size=(128, 128, 128), random_size=False),
        transforms.ScaleIntensityRangePercentilesd(keys=["image", "frangi"], lower=0, upper=100, b_min=0, b_max=1),
    ]
)


def get_tof_dataset(data_path, transform):
    data = []
    sublist = os.listdir(data_path)
    for subject in sublist:
        sub_path = os.path.join(data_path, subject)
        
        image = os.path.join(sub_path, f"img.nii.gz")
        frangi = os.path.join(sub_path, f"frangi.nii.gz")
    
        label = os.path.join(sub_path, f"th.nii.gz")
        if os.path.isfile(frangi) and os.path.isfile(label):
            data.append({"image":image, "frangi": frangi, "mask":label})
                    
    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)


class CustomBase(Dataset):
    def __init__(self,data_path, transform):
        super().__init__()
        self.data = get_tof_dataset(data_path, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class CustomTrain(CustomBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, transform=TOF_transforms)


class CustomTest(CustomBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, transform=TOF_testforms)
