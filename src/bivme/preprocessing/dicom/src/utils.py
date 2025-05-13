import os
import numpy as np
import nibabel as nib
import cv2
import scipy.ndimage as ndimage

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
    

def write_nifti(slice_id, pixel_array, pixel_spacing, input_folder, view):
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

def resample_img(dst, view, series, num_phases, my_logger):
    # Load 3D nifti
    img = nib.load(os.path.join(dst, 'images', view, '{}_3d_{}_0000.nii.gz'.format(view, series)))
    img_array = img.get_fdata()

    # Need to resample last dimension to num_phases
    current_phases = img_array.shape[-1]

    # Apply spline interpolation in the temporal dimension
    new_img_array = ndimage.zoom(img_array, (1, 1, num_phases/current_phases), order=3) # Order 3 is cubic spline
    new_img_array = new_img_array.astype(np.uint8)

    # Save as 3D nii
    affine = img.affine
    new_nii = nib.Nifti1Image(new_img_array, affine)
    nib.save(new_nii, os.path.join(dst, 'images', view, '{}_3d_{}_0000.nii.gz'.format(view, series)))

def resample_seg(dst, view, series, num_phases, my_logger):
    # Load 3D nifti
    seg = nib.load(os.path.join(dst, 'segmentations', view, '{}_3d_{}.nii.gz'.format(view, series)))
    seg_array = seg.get_fdata()

    # Need to resample last dimension to num_phases
    current_phases = seg_array.shape[-1]

    # Apply spline interpolation in the temporal dimension
    new_seg_array = ndimage.zoom(seg_array, (1, 1, num_phases/current_phases), order=0) # Order 0 is nearest neighbour
    new_seg_array = new_seg_array.astype(np.uint8)

    # Save as 3D nii
    affine = seg.affine
    new_nii = nib.Nifti1Image(new_seg_array, affine)
    nib.save(new_nii, os.path.join(dst, 'segmentations', view, '{}_3d_{}.nii.gz'.format(view, series)))

def clean_text(string):

    # clean and standardize text descriptions, which makes searching files easier

    forbidden_symbols = ["*", ".", ",", "\"", "\\", "/", "|", "[", "]", ":", ";", " "]
    for symbol in forbidden_symbols:
        string = string.replace(symbol, "")  # replace all bad symbols

    return string.lower()

def from_2d_to_3d(
    p2, image_orientation, image_position, pixel_spacing
):
    """# Convert indices of a pixel in a 2D image in space to 3D coordinates.
    #	Inputs
    #		image_orientation
    #		image_position
    #		pixel_spacing
    #		subpixel_resolution
    #	Outputs
    #		P3:  3D points
    """
    # if points2D.
    points2D = np.array(p2)

    S = np.eye(4)
    S[0, 0] = pixel_spacing[1]
    S[1, 1] = pixel_spacing[0]
    S = np.matrix(S)

    R = np.identity(4)
    R[0:3, 0] = image_orientation[
        0:3
    ]  # col direction, i.e. increases with row index i
    R[0:3, 1] = image_orientation[
        3:7
    ]  # row direction, i.e. increases with col index j
    R[0:3, 2] = np.cross(R[0:3, 0], R[0:3, 1])

    T = np.identity(4)
    T[0:3, 3] = image_position

    F = np.identity(4)
    F[0:1, 3] = -0.5

    T = np.dot(T, R)
    T = np.dot(T, S)
    Transformation = np.dot(T, F)

    pts = np.ones((len(points2D), 4))
    pts[:, 0:2] = points2D
    pts[:, 2] = [0] * len(points2D)
    pts[:, 3] = [1] * len(points2D)

    Px = np.dot(Transformation, pts.T)
    p3 = Px[0:3, :] / (np.vstack((Px[3, :], np.vstack((Px[3, :], Px[3, :])))))
    p3 = p3.T

    return p3[0, 0], p3[0, 1], p3[0, 2]