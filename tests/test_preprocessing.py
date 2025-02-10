import os
import PIL
import pandas as pd 
import shutil
from loguru import logger
from bivme import TEST_RESOURCE_DIR

from bivme.preprocessing.dicom.src.viewselection import ViewSelector

def test_viewselection():
    test_csv_path = ''
    test_src = os.path.join(TEST_RESOURCE_DIR, 'viewselection_data', 'dicoms')
    test_dst = os.path.join(TEST_RESOURCE_DIR, 'viewselection_data', 'output-pngs', 'patient1')
    test_model = ''

    os.makedirs(test_dst, exist_ok=True)

    viewSelector = ViewSelector(test_src, test_dst, test_model, csv_path=test_csv_path, my_logger=logger)
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