from bivme.analysis import compute_volume
from pytest import approx
from bivme.analysis.compute_volume import find_volume
from bivme.analysis.compute_wall_thickness import find_wall_thickness
from bivme import MODEL_RESOURCE_DIR, TEST_RESOURCE_DIR
import csv
import os
import pandas as pd
from pathlib import Path
import numpy as np
import shutil
def test_compute_volume():
    model_file = TEST_RESOURCE_DIR / 'template' / 'template_model_frame_001.txt'
    output_file = 'test_lvrv_volumes.csv'

    fieldnames = ['name', 'frame', 'lv_vol', 'lvm', 'rv_vol', 'rvm', 'lv_epivol', 'rv_epivol']
    with open(output_file, 'w', newline='') as f:
        # create output file and write header
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    find_volume('template_mesh', model_file, output_file, MODEL_RESOURCE_DIR, 6)

    df = pd.read_csv(output_file)
    assert df['lv_vol'][0] == approx(0.074302)  # ground truth values
    assert df['rv_vol'][0] == approx(0.08629507)
    assert df['lv_epivol'][0] == approx(0.158451)
    assert df['rv_epivol'][0] == approx(0.116371)
    assert df['rvm'][0] == approx(0.03158)
    assert df['lvm'][0] == approx(0.088356)

    os.remove('test_lvrv_volumes.csv')

def test_compute_wall_thickness():
    biv_model_file = TEST_RESOURCE_DIR / 'template' / 'template_model_frame_001.txt'

    output_folder = TEST_RESOURCE_DIR / "wall_thickness_output"
    output_folder.mkdir(exist_ok=True)

    output_folder_patient = output_folder / 'template'
    output_folder_patient.mkdir(exist_ok=True)

    find_wall_thickness('template', biv_model_file, output_folder_patient, MODEL_RESOURCE_DIR, 0.1, False)

    case_list = os.listdir(Path(output_folder_patient))
    case_dirs = [case for case in case_list]

    assert len(case_dirs) == 2

    shutil.rmtree(output_folder_patient)

    output_folder_patient.mkdir(exist_ok=True)
    find_wall_thickness('template', biv_model_file, output_folder_patient, MODEL_RESOURCE_DIR, 0.1, True)

    case_list = os.listdir(Path(output_folder_patient))
    case_dirs = [case for case in case_list]

    assert len(case_dirs) == 6

    shutil.rmtree(output_folder_patient)
    shutil.rmtree(output_folder)