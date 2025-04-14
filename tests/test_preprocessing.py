import os
import numpy as np
import nibabel as nib
import PIL
import pandas as pd 
import shutil
from loguru import logger
from bivme import TEST_RESOURCE_DIR

from bivme.preprocessing.dicom.src.viewselection import ViewSelector
from bivme.preprocessing.dicom.src.utils import write_nifti
from bivme.preprocessing.dicom.src.utils import write_sliceinfofile
from bivme.preprocessing.dicom.generate_contours import generate_contours
from bivme.preprocessing.dicom.export_guidepoints import export_guidepoints

def test_viewselection(): # This test checks whether the dicom-png conversion and csv generation prior to view prediction works as expected
    test_csv_path = ''
    test_src = os.path.join(TEST_RESOURCE_DIR, 'viewselection_data', 'dicoms')
    test_dst = os.path.join(TEST_RESOURCE_DIR, 'viewselection_data', 'output-pngs', 'patient1')
    test_model = ''

    os.makedirs(test_dst, exist_ok=True)

    viewSelector = ViewSelector(test_src, test_dst, test_model, type='image', csv_path=test_csv_path, my_logger=logger)
    viewSelector.prepare_data_for_prediction()

    reference_root = os.path.join(TEST_RESOURCE_DIR, 'viewselection_data', 'reference-pngs', 'patient1', 'unsorted')
    reference_image_paths = [os.path.join(reference_root, x) for x in os.listdir(reference_root)] # sorry for path gore
    test_image_paths = [os.path.join(test_dst, 'view-classification', 'unsorted', x) for x in os.listdir(os.path.join(test_dst, 'view-classification', 'unsorted'))]

    reference_images = [PIL.Image.open(x) for x in reference_image_paths]
    test_images = [PIL.Image.open(x) for x in test_image_paths]

    assert len(reference_images) == len(test_images), 'Number of images do not match.'
    for i in range(len(reference_images)):
        assert reference_images[i].size == test_images[i].size, f'Image {i} size does not match.'

    # Compare contents of the csv files
    reference_csv = os.path.join(TEST_RESOURCE_DIR, 'viewselection_data', 'reference-pngs', 'patient1', 'test_annotations.csv')
    test_csv = os.path.join(test_dst, 'view-classification', 'test_annotations.csv')

    reference_df = pd.read_csv(reference_csv)
    test_df = pd.read_csv(test_csv)

    assert reference_df.equals(test_df), 'Dataframes do not match.'

    # Clean up
    # Close all images
    for img in test_images:
        img.close()
    shutil.rmtree(test_dst)

def test_writenifti():
    root = os.path.join(TEST_RESOURCE_DIR, 'writenifti_data')

    # Set up dummy data
    slice_id = 1
    view = "SAX"
    pixel_spacing = [1, 1]
    
    # Create a 3D image
    img = np.zeros((27, 256, 256))
    initial_x, initial_y = 50, 200
    for i in range(27):
        for j in range(256):
            for k in range(256):
                if initial_x+(2*i) < j < initial_y-(2*i) and initial_x+(2*i) < k < initial_y-(2*i) and j < k:
                    img[i, j, k] = 1
        img[i] = img[i] * 255
        img[i] = img[i].astype(np.uint8)
    
    test_dst = os.path.join(root, 'output', 'patient1')
    os.makedirs(test_dst, exist_ok=True)

    os.makedirs(os.path.join(test_dst, view), exist_ok=True)

    write_nifti(slice_id, img, pixel_spacing, test_dst, view, '3d') # Testing 3d version

    # Find the generated nifti file
    nifti_path = os.path.join(test_dst, view, f'{view}_3d_{slice_id}_0000.nii.gz')
    assert os.path.exists(nifti_path), 'Nifti file not found.'

    # Compare to reference
    reference_path = os.path.join(root, 'reference-nifti', 'patient1', view, f'{view}_3d_{slice_id}_0000.nii.gz')
    reference_nifti = nib.load(reference_path)
    test_nifti = nib.load(nifti_path)

    assert reference_nifti.get_fdata().shape == test_nifti.get_fdata().shape, 'Nifti shape does not match.'

    assert np.array_equal(reference_nifti.get_fdata(), test_nifti.get_fdata()), 'Nifti data does not match.'

    # Clean up
    shutil.rmtree(test_dst)

def test_contouring():
    root = os.path.join(TEST_RESOURCE_DIR, 'contouring_data')

    # Set up dummy data
    case = 'patient1'
    slice_id = 1
    filename = f"dummy.dcm"
    view = "SAX"
    pixel_spacing = [1, 1]
    image_position_patient = [0, 0, 0]
    image_orientation_patient = [1, 0, 0, 0, 1, 0]
    img = np.zeros((27, 256, 256))

    output = [slice_id, filename, view, image_position_patient, image_orientation_patient, pixel_spacing, img]
    slice_info_df = pd.DataFrame([output], columns=['Slice ID', 'File', 'View', 'ImagePositionPatient', 'ImageOrientationPatient', 'Pixel Spacing', 'Img'])

    test_dst = os.path.join(root, 'patient1')
    test_output = os.path.join(root, 'output', 'patient1')
    os.makedirs(test_output, exist_ok=True)

    # Write slice info file
    slice_mapping = write_sliceinfofile(test_dst, slice_info_df)

    # Try finding the slice info file
    slice_info_file = os.path.join(test_dst, 'SliceInfoFile.txt')
    assert os.path.exists(slice_info_file), 'SliceInfoFile not generated.'

    # Generate contours
    slice_dict = generate_contours(case, test_dst, slice_info_df, 27, '3d', logger)

    assert len(slice_dict.keys()) == 1, 'Contours not generated.'

    # Export contours as GP files
    export_guidepoints(case, test_dst, os.path.join(root,'output'), slice_dict, slice_mapping)

    # Find the GP files
    gp_files = [f for f in os.listdir(test_output) if 'GPFile' in f]
    assert len(gp_files) == 27, 'Not all GP files were generated.'

    # Clean up
    shutil.rmtree(test_output)