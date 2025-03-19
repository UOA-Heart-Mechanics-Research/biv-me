import os
import numpy as np
import nibabel as nib
import cv2

def write_sliceinfofile(dst, slice_info_df):
    # Calculate a slice mapping (reformat to 1-numslices)
    slice_mapping = {}
    for i, row in slice_info_df.iterrows():
        slice_mapping[row['Slice ID']] = i+1
        
    # write to slice info file
    with open(os.path.join(dst, 'SliceInfoFile.txt'), 'w') as f:
        for i, row in slice_info_df.iterrows():
            sliceID = slice_mapping[row['Slice ID']]
            file = row['File']
            file = os.path.basename(file)
            view = row['View']
            imagePositionPatient = row['ImagePositionPatient']
            imageOrientationPatient = row['ImageOrientationPatient']
            pixelSpacing = row['Pixel Spacing']
            
            f.write('{}\t'.format(file))
            f.write('sliceID: \t')
            f.write('{}\t'.format(sliceID))
            f.write('ImagePositionPatient\t')
            f.write('{}\t{}\t{}\t'.format(imagePositionPatient[0], imagePositionPatient[1], imagePositionPatient[2]))
            f.write('ImageOrientationPatient\t')
            f.write('{}\t{}\t{}\t{}\t{}\t{}\t'.format(imageOrientationPatient[0], imageOrientationPatient[1], imageOrientationPatient[2],
                                                imageOrientationPatient[3], imageOrientationPatient[4], imageOrientationPatient[5]))
            f.write('PixelSpacing\t')
            f.write('{}\t{}\n'.format(pixelSpacing[0], pixelSpacing[1]))
    
    return slice_mapping
    

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

def resample_seg(dst, view, series, num_phases, my_logger):
    # Load 3D nifti
    seg = nib.load(os.path.join(dst, 'segmentations', view, '{}_3d_{}.nii.gz'.format(view, series)))
    seg_array = seg.get_fdata()

    # Need to resample last dimension to num_phases
    current_phases = seg_array.shape[-1]

    # Interpolate each pixel in the last dimension
    new_seg_array = np.zeros((seg_array.shape[0], seg_array.shape[1], num_phases))
    label_classes = np.unique(seg_array)
    label_arrays = np.zeros((seg_array.shape[0], seg_array.shape[1], num_phases, len(label_classes)))

    # Perform interpolation seperately for each label class
    for l in label_classes:
        l = int(l)
        if l == 0:
            continue
        new_label_array = np.zeros((seg_array.shape[0], seg_array.shape[1], num_phases))
        label_array = (seg_array == l).astype(np.uint8)
        for i in range(label_array.shape[0]):
            for j in range(label_array.shape[1]):
                new_label_array[i, j, :] = np.interp(np.linspace(0, current_phases-1, num_phases), np.arange(current_phases), label_array[i, j]) # TODO: Pretty rudimentary, but should work. Maybe revisit later
        
        # Threshold 
        new_label_array[new_label_array > 0.5] = l
        new_label_array[new_label_array <= 0.5] = 0

        label_arrays[:, :, :, l] = new_label_array
    
    # Reassemble
    for l in label_classes: # TODO: Not ideal because there's no sense of priority, might want to order label classes by reverse importance (e.g. LV overwrites LA)
        l = int(l)
        if l == 0:
            continue
        new_seg_array[label_arrays[:, :, :, l] == l] = l

    new_seg_array = new_seg_array.astype(np.uint8)

    # Save as 3D nii
    affine = seg.affine
    new_nii = nib.Nifti1Image(new_seg_array, affine)
    nib.save(new_nii, os.path.join(dst, 'segmentations', view, '{}_3d_{}.nii.gz'.format(view, series)))

    pass