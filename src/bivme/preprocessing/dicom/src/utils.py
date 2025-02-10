import os
import numpy as np
import nibabel as nib
import cv2

def write_nifti(slice_id, pixel_array, pixel_spacing, input_folder, view, version):
    if version == '2d':
        for frame, img in enumerate(pixel_array):
            img = img.astype(np.float32)
            # Transpose
            img = np.transpose(img)

            affine = np.eye(4)
            affine[0, 0] = pixel_spacing[0]
            affine[1, 1] = pixel_spacing[1]
            img_nii = nib.Nifti1Image(img, affine)
            nib.save(img_nii, os.path.join(input_folder, view, '{}_2d_{}_{:03}_0000.nii.gz'.format(view, slice_id, frame)))

        rescale_factor = 1 # Dummy value for now
            
    elif version == '3d':
        img = pixel_array.astype(np.float32)
        # Transpose so that the last dimension is the number of frames
        img = np.transpose(img, (1, 2, 0))
        # Transpose width and height
        img = np.transpose(img, (1, 0, 2))

        # Pad to square
        max_dim = max(img.shape)
        pad = [(0, 0), (0, 0), (0, 0)]
        pad[0] = (0, max_dim - img.shape[0])
        pad[1] = (0, max_dim - img.shape[1])
        img = np.pad(img, pad, mode='constant', constant_values=0)

        # Pad to 256x256, or resize to 256x256 if it's larger
        current_dims = img.shape
        if current_dims[0] < 256 or current_dims[1] < 256:
            # Pad to 256x256, adding in opposite corner to origin
            pad = [(0, 256 - current_dims[0]), (0, 256 - current_dims[1]), (0, 0)]
            img = np.pad(img, pad, mode='constant', constant_values=0)
            rescale_factor = 1

        elif current_dims[0] > 256 or current_dims[1] > 256:
            # Resize to 256x256
            rescale_factor = max(current_dims[0], current_dims[1]) / 256 # Need to change pixel spacing accordingly
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        
        else:
            rescale_factor = 1

        # Remap pixel values to 0-255
        img = img - np.min(img)
        img = img / np.max(img) * 255
        img = img.astype(np.uint8)

        affine = np.eye(4) # Default pixel spacing is 1,1,1. This is what the segmentation model expects

        img_nii = nib.Nifti1Image(img, affine)
        nib.save(img_nii, os.path.join(input_folder, view, '{}_3d_{}_0000.nii.gz'.format(view, slice_id)))

    return rescale_factor
