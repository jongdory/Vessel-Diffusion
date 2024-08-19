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
import numpy as np
import nibabel as nib
import argparse
from skimage.measure import label, regionprops
from sklearn.mixture import GaussianMixture
from scipy.ndimage import binary_dilation, center_of_mass

# argument parser
def get_args():
    parser = argparse.ArgumentParser(description='GMM')
    parser.add_argument('-d', '--data_path', type=str, default='data')
    parser.add_argument('-n', '--n_components', type=int, default=6)
    parser.add_argument('-e', '--eps', type=float, default=0.05)
    parser.add_argument('-k', '--kernel_size', type=int, default=7)
    args = parser.parse_args()
    return args


# calculate threshold using GMM
def get_threshold(arr, n_components):
    arr = arr[::8,::8,::8]
    data = arr[arr>=0.05]
    data = data.reshape(-1,1)
    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
    gmm.fit(data)

    labels = gmm.predict(data)
    
    arr = []
    for i in range(n_components):
        arr.append(data[labels == i].min())

    threshold = np.array(arr).max()

    return threshold

def distance(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def get_label(arr, threshold, label_arr=None, kernel_size=7):
    # --- get threshold label ---
    mask = np.zeros_like(arr)
    mask[arr > threshold] = 1
    masses = []

    # devide mass
    labeledImage = label(mask!=0)

    # get mass size
    areas = [r.area for r in regionprops(labeledImage)]
    sorted_areas = sorted(areas, reverse=True)
    
    for area in sorted_areas:
        if area < 3000: break
        index = areas.index(area) + 1
        zeros = np.zeros_like(mask)
        zeros[labeledImage==index] = 1
        
        if label_arr is not None:
            if np.sum(zeros * label_arr) > 0:
                masses.append(areas.index(area) + 1)
        else:
            masses.append(areas.index(area) + 1)
            break
            
    mask = np.zeros_like(mask)
    for massIndex in masses:
        mask[labeledImage==massIndex] = 1

    if label_arr is None:
        kernel = np.ones((kernel_size, kernel_size, kernel_size), np.uint8)
        mask = binary_dilation(mask, iterations=4, structure=kernel).astype(np.uint8)

    return mask

def save_label(image, mask, save_path):
    mask = mask.astype(np.uint8)
    nii = nib.Nifti1Image(mask, image.affine, image.header)
    nib.save(nii, save_path)

def maybe_mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def main():
    args = get_args()
    
    data_path = args.data_path
    n_components = args.n_components
    eps = args.eps

    image = nib.load(f'{data_path}/normalized.nii.gz')
    arr = image.get_fdata()
    th = get_threshold(arr, n_components=n_components)
    print(data_path, th)
    low_th, high_th = th-eps, th+eps
    maybe_mkdir(f'{data_path}/labels')

    label_mass = get_label(arr, th, kernel_size=args.kernel_size)
    
    save_label(image, label_mass, f'{data_path}/labels/mass.nii.gz')

    th_label = get_label(arr, th, label_mass)
    high_th_label = get_label(arr, high_th, label_mass)
    low_th_label = get_label(arr, low_th, label_mass)

    save_label(image, th_label, f'{data_path}/labels/th.nii.gz')
    save_label(image, high_th_label, f'{data_path}/labels/high_th.nii.gz')
    save_label(image, low_th_label, f'{data_path}/labels/low_th.nii.gz')


if __name__ == '__main__':
    main()