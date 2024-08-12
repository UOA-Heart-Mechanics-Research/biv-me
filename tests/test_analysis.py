from bivme.analysis import compute_volume
from pytest import approx
from bivme.analysis.compute_volume import find_volume
from bivme import MODEL_RESOURCE_DIR, TEST_RESOURCE_DIR
import csv
import os
import pandas as pd

def test_compute_volume():
    model_file = TEST_RESOURCE_DIR / 'case_1' / 'case_1_model_frame_001.txt'
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
    assert df['lv_epivol'][0] == approx(0.162056)
    assert df['rv_epivol'][0] == approx(0.112766)
    assert df['rvm'][0] == approx(0.027795)
    assert df['lvm'][0] == approx(0.092142)

    os.remove('test_lvrv_volumes.csv')
