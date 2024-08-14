import numpy as np
from bivme.fitting.perform_fit import perform_fitting
from bivme import TEST_RESOURCE_DIR
import shutil

def test_performed_fit():

    patient_name = 'patient_1_gpdata'
    gp_file = TEST_RESOURCE_DIR / patient_name
    test_files = ['patient_1_gpdata_model_frame_000.txt', 'patient_1_gpdata_model_frame_001.txt']
    output_dir = TEST_RESOURCE_DIR / 'output'
    if not output_dir.exists():
        output_dir.mkdir()
    perform_fitting(gp_file, output_dir)

    for test_file in test_files:
        gt = np.loadtxt(TEST_RESOURCE_DIR / patient_name / test_file, delimiter=',', skiprows=1, usecols=[0, 1, 2]).astype(float)

        # check .txt file was created
        test_model = TEST_RESOURCE_DIR / 'output' / patient_name / test_file
        assert test_model.exists(), \
            f"No model created!"

        test = np.loadtxt(test_model, delimiter=',', skiprows=1, usecols=[0, 1, 2]).astype(float)
        assert np.array_equal(gt, test)

    shutil.rmtree(output_dir)


