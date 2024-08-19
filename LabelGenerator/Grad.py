# Copyright (c) [2024] [Jonghun Kim]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import nibabel as nib
from skimage import img_as_ubyte
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize_3d
from scipy.ndimage import binary_erosion, binary_dilation, zoom
import logging
import argparse

# argument parser
def get_args():
    parser = argparse.ArgumentParser(description='GMM')
    parser.add_argument('-d', '--data_path', type=str, default='data')
    parser.add_argument('-n', '--n_components', type=int, default=6)
    parser.add_argument('-e', '--eps', type=float, default=0.05)
    parser.add_argument('-k', '--kernel_size', type=int, default=7)
    args = parser.parse_args()
    return args

# Calculate the derivative using NumPy's gradient function
def derivative(arr):
    arr = resize_256(arr)
    hist, bin_edges = np.histogram(arr[arr>0.1], bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    deriv = np.gradient(hist, bin_centers)

    return deriv, bin_centers

# Find the threshold value low (-0.005), high (-0.15)
def get_threshold(deriv, bin_centers, low=-0.1, high = -0, scale=1e5): # scale=1e8
    thresholds_indexs = np.where((deriv > low * scale) & (deriv < high * scale))[0]
    min_index = np.where(deriv == deriv.min())[0][0]

    threshold_index = thresholds_indexs[thresholds_indexs > min_index][0]
    threshold= bin_centers[threshold_index]

    return threshold

def resize_256(img_data, new_size=(256, 256, 256)):
    original_size = img_data.shape
    scale_factors = [new_dim / old_dim for new_dim, old_dim in zip(new_size, original_size)]
    
    return zoom(img_data, scale_factors, order=3)
    

def save_label(image, arr, label_arr, logger, threshold, save_path):
    # --- get threshold label ---
    mask = np.zeros_like(arr)
    mask[arr > threshold] = 1

    kernel = np.ones((5,5,5), np.uint8)
    label_arr = binary_dilation(binary_erosion(label_arr, structure=kernel), structure=kernel)

    masses = []

    # devide mass
    labeledImage = label(mask!=0)

    # get mass size
    areas = [r.area for r in regionprops(labeledImage)]
    sorted_areas = sorted(areas, reverse=True)
    
    for area in sorted_areas:
        if area < 10000: break
        index = areas.index(area) + 1
        zeros = np.zeros_like(mask)
        zeros[labeledImage==index] = 1
        # masses.append(areas.index(area) + 1)
        if np.sum(zeros * label_arr) > 0:
            masses.append(areas.index(area) + 1)
            
    mask = np.zeros_like(mask)
    for massIndex in masses:
        mask[labeledImage==massIndex] = 1

    # size of mass
    logger.info(f"Threshold: {threshold}, Masses: {sorted_areas[:5]}")

    nii = nib.Nifti1Image(mask, image.affine, image.header)
    nib.save(nii, save_path)

    return mask

def maybe_mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', 
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler() 
        ]
    )

    logger = logging.getLogger(__name__)

    args = get_args()
    data_path = args.data_path

    print(data_path)
    subject = data_path.split('/')[-1]
    logger.info(f"{subject} is started")
        
    image = nib.load(f'{data_path}/normalized.nii.gz')
    maybe_mkdir(f'{data_path}/label')
    arr = image.get_fdata()
    deriv, bin_centers = derivative(arr)
    threshold = get_threshold(deriv, bin_centers)
    print(data_path, threshold)

    label_arr = save_label(image, arr, np.ones_like(arr), logger, threshold + 0.07, f'{data_path}/label/high_th_0.07.nii.gz')
    
    print(f"Threshold: {threshold}")
    logger.info(f"Threshold: {threshold}")

    save_label(image, arr, label_arr, logger, threshold, f'{data_path}/label/th.nii.gz')
    save_label(image, arr, label_arr, logger, threshold - 0.01, f'{data_path}/label/low_th_0.01.nii.gz')
    save_label(image, arr, label_arr, logger, threshold - 0.03, f'{data_path}/label/low_th_0.03.nii.gz')
    save_label(image, arr, label_arr, logger, threshold - 0.05, f'{data_path}/label/low_th_0.05.nii.gz')
    save_label(image, arr, label_arr, logger, threshold - 0.07, f'{data_path}/label/low_th_0.07.nii.gz')
    save_label(image, arr, label_arr, logger, threshold - 0.1, f'{data_path}/label/low_th_0.1.nii.gz')
    save_label(image, arr, label_arr, logger, threshold + 0.01, f'{data_path}/label/high_th_0.01.nii.gz')
    save_label(image, arr, label_arr, logger, threshold + 0.03, f'{data_path}/label/high_th_0.03.nii.gz')
    save_label(image, arr, label_arr, logger, threshold + 0.05, f'{data_path}/label/high_th_0.05.nii.gz')
    save_label(image, arr, label_arr, logger, threshold + 0.07, f'{data_path}/label/high_th_0.07.nii.gz')
    save_label(image, arr, label_arr, logger, threshold + 0.1, f'{data_path}/label/high_th_0.1.nii.gz')

    logger.info(f"{subject} is finished")


if __name__ == '__main__':
    main()