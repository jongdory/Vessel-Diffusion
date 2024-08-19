#!/bin/bash

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
#

input_directory="./input"
output_directory="./output"

# GMM parameters
n_components=6
eps=0.05
kernel_size=7

# Iterate over all subdirectories within the input directory
for subject in "$input_directory"/*/; do
    start=$(date +%s)

    subject_name=$(basename "$subject")

    echo "Processing subject: $subject_name"

    # Create the output directory (to store results for each subdirectory)
    input_dir="$input_directory/$subject_name"
    output_dir="$output_directory/$subject_name"
    mkdir -p "$output_dir"
      
    input_image="$input_dir/image.nii"
    output_image="$output_dir/corrected.nii.gz"
    bias_field_image="$output_dir/BiasField.nii.gz"

    # Intensity Normalization
    normalized_image="$output_dir/normalized.nii.gz"
    ImageMath 3 "$normalized_image" Normalize "$input_image"

    # Bias Field Correction
    N4BiasFieldCorrection -d 3 -v 1 -s 4 -b [ 180 ] -c [ 50x50x50x50, 0.0 ] -i "$normalized_image" -o "$output_image"

    # Denoising
    denoised_image="$output_dir/denoised.nii.gz"
    # DenoiseImage -d 3 -i "$normalized_image" -n Rician -s 2 -p 1 -r 3 -o "$denoised_image" -v 1
    DenoiseImage -d 3 -i "$output_image" -n Rician -s 2 -p 1 -r 3 -o "$denoised_image" -v 1

    # Intensity Normalization
    normalized_image="$output_dir/normalized.nii.gz"
    ImageMath 3 "$normalized_image" Normalize "$denoised_image"

    # Remove Files
    # rm "$output_image"
    # rm "$bias_field_image"
    # rm "$denoised_image"

    # Get label automatically using GMM
    python3 GMM.py -d "$output_dir" -n "$n_components" -e "$eps" -k "$kernel_size"

    endt=$(date +%s)
    runtime=$((endt-start))
    echo "Preprocessing took $runtime seconds"
done

echo "Preprocessing completed"
