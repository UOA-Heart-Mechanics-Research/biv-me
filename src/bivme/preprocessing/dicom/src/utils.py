import os
import numpy as np
import nibabel as nib
import cv2
import scipy.ndimage as ndimage


def write_sliceinfofile(dst, slice_info_df):
    # write to slice info file
    with open(os.path.join(dst, 'SliceInfoFile.txt'), 'w') as f:
        for i, row in slice_info_df.iterrows():
            sliceID = int(row['Slice ID'])
            file = row['File']
            file = os.path.basename(file)
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
    

def write_nifti(slice_id, pixel_array, pixel_spacing, input_folder, view):
    img = pixel_array.astype(np.float32)
    # Transpose so that the last dimension is the number of frames
    img = np.transpose(img, (1, 2, 0))

    ## Method for Datasets 260-264
    # Resize to 1x1 mm pixel spacing
    current_spacing = pixel_spacing
    if current_spacing[0] != 1 or current_spacing[1] != 1:
        new_img = np.zeros((int(img.shape[0] * pixel_spacing[0]), int(img.shape[1] * pixel_spacing[1]), img.shape[2]), dtype=img.dtype)
        new_shape = new_img.shape
        if new_img.shape[0] < 256 or new_img.shape[1] < 256:
            ratio = max(256 / new_img.shape[0], 256 / new_img.shape[1])
            new_shape = (int(np.ceil(new_img.shape[0] * ratio)), int(np.ceil(new_img.shape[1] * ratio)), new_img.shape[2])
            new_img = np.zeros(new_shape, dtype=img.dtype)
        else:
            ratio = 1

        for frame in range(img.shape[2]):
            new_img[:, :, frame] = cv2.resize(img[:, :, frame], (new_img.shape[1], new_img.shape[0]), interpolation=cv2.INTER_CUBIC)

    elif img.shape[0] < 256 or img.shape[1] < 256:
        # Resize smallest dimension to 256, keeping aspect ratio
        ratio = max(256 / img.shape[0], 256 / img.shape[1])
        new_shape = (int(np.ceil(img.shape[0] * ratio)), int(np.ceil(img.shape[1] * ratio)), img.shape[2])
        new_img = np.zeros(new_shape, dtype=img.dtype)
        
        for frame in range(img.shape[2]):
            new_img[:, :, frame] = cv2.resize(img[:, :, frame], (new_img.shape[1], new_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    else:
        ratio = 1 # The 'ratio' is the new pixel spacing for the resized images. Usually 1, unless resized images are less than 256x256, in which case they smaller than 1
        new_img = img

    rescale_factor = 1/(current_spacing[0] * ratio)

    # Crop
    img = new_img
    orig_img = img.copy()
    # Apply center crop of 256x256
    target_size = 256
    original_shape = img.shape
    new_shape = list(original_shape)

    if original_shape[0] < original_shape[1]:
        new_shape[0] = target_size
        new_shape[1] = target_size
    else:
        new_shape[0] = target_size
        new_shape[1] = target_size

    new_img = np.zeros((int(new_shape[0]), int(new_shape[1]), int(new_shape[2])))
    start_x = (original_shape[0] - target_size) // 2
    start_y = (original_shape[1] - target_size) // 2

    for i in range(img.shape[2]):
        slice_img = img[:,:,i]
        cropped_slice = slice_img[start_x:start_x+target_size, start_y:start_y+target_size]
        new_img[:,:,i] = cropped_slice

    img = new_img

    ## Method for Datasets 260-264
    # Normalise intensity values
    new_img = np.zeros_like(img)
    for frame in range(img.shape[2]):
        slice_data = img[:, :, frame]

        min_value = min(0, np.min(slice_data))
        slice_data += abs(min_value)  # Shift to make all values positive because CLAHE requires uint type (0-)

        # Apply CLAHE
        clahe = cv2.createCLAHE(tileGridSize=(1,1)) 
        slice_data_clahe = clahe.apply(slice_data.astype(np.uint16))
        slice_data = slice_data_clahe

        # # Apply 1-99 percentile clipping
        p1 = np.percentile(slice_data, 0.5)
        p99 = np.percentile(slice_data, 99.5)
        slice_data = np.clip(slice_data, p1, p99)

        # Z-score normalisation (unmasked)
        mean_intensity = np.mean(slice_data)
        std_intensity = np.std(slice_data)
        slice_data_normalised = (slice_data - mean_intensity) / std_intensity
        new_img[:, :, frame] = slice_data_normalised

    img = new_img

    affine = np.eye(4) # Default pixel spacing is 1,1,1. This is what the segmentation model expects

    orig_img_nii = nib.Nifti1Image(orig_img, affine)
    # Make original image folder if doesn't exist
    os.makedirs(os.path.join(input_folder, view, 'resized'), exist_ok=True)
    nib.save(orig_img_nii, os.path.join(input_folder, view, 'resized', '{}_3d_{}_0000.nii.gz'.format(view, slice_id)))

    img_nii = nib.Nifti1Image(img, affine)
    # Make processed image folder if doesn't exist
    os.makedirs(os.path.join(input_folder, view, 'resized-cropped-normalised'), exist_ok=True)
    nib.save(img_nii, os.path.join(input_folder, view, 'resized-cropped-normalised', '{}_3d_{}_0000.nii.gz'.format(view, slice_id)))

    return rescale_factor


def resample_img(dst, view, series, num_phases, my_logger):
    # Load 3D nifti
    img = nib.load(os.path.join(dst, 'images', view, 'resized', '{}_3d_{}_0000.nii.gz'.format(view, series)))
    img_array = img.get_fdata()

    # Need to resample last dimension to num_phases
    current_phases = img_array.shape[-1]

    # Apply spline interpolation in the temporal dimension
    new_img_array = ndimage.zoom(img_array, (1, 1, num_phases/current_phases), order=3) # Order 3 is cubic spline

    # Save as 3D nii
    affine = img.affine
    new_nii = nib.Nifti1Image(new_img_array, affine)
    nib.save(new_nii, os.path.join(dst, 'images', view, 'resized', '{}_3d_{}_0000.nii.gz'.format(view, series)))


def resample_seg(dst, view, series, num_phases, my_logger):
    # Load 3D nifti
    seg = nib.load(os.path.join(dst, 'segmentations', view, 'uncropped', '{}_3d_{}.nii.gz'.format(view, series)))
    seg_array = seg.get_fdata()

    # Need to resample last dimension to num_phases
    current_phases = seg_array.shape[-1]

    # Apply spline interpolation in the temporal dimension
    new_seg_array = ndimage.zoom(seg_array, (1, 1, num_phases/current_phases), order=0) # Order 0 is nearest neighbour
    new_seg_array = new_seg_array.astype(np.uint8)

    # Save as 3D nii
    affine = seg.affine
    new_nii = nib.Nifti1Image(new_seg_array, affine)

    nib.save(new_nii, os.path.join(dst, 'segmentations', view, 'uncropped', '{}_3d_{}.nii.gz'.format(view, series)))


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


def plane_intersect(a, b):
    """
    a, b   4-tuples/lists
           Ax + By +Cz + D = 0
           A,B,C,D in order  

    output: 2 points on line of intersection, np.arrays, shape (3,)
    """
    a_vec, b_vec = np.array(a[:3]), np.array(b[:3])

    aXb_vec = np.cross(a_vec, b_vec)

    A = np.array([a_vec, b_vec, aXb_vec])
    d = np.array([-a[3], -b[3], 0.]).reshape(3,1)

    try:
        p_inter = np.linalg.solve(A, d).T
    except np.linalg.LinAlgError:
        # If the planes are parallel or coincident, return NaN
        return np.nan, np.nan

    return p_inter[0], (p_inter + aXb_vec)[0]


def fft(curve, harmonic_divisor=4):
    fft_volume = np.fft.rfft(curve)
    fft_volume[int(np.floor(len(curve)/harmonic_divisor)):] = 0
    curve_filtered = np.fft.irfft(fft_volume)

    return curve_filtered


def apply_fft(curve, harmonic_divisor=4):
    if len(curve) % 2 != 0:
        # Append the first value to the end
        curve = np.append(curve, curve[0])
        curve_filtered = fft(curve, harmonic_divisor=harmonic_divisor)
        curve_filtered = curve_filtered[:-1]  # Remove the last value to keep the length consistent
    else:
        curve_filtered = fft(curve, harmonic_divisor=harmonic_divisor)
        
    return curve_filtered