import numpy as np
from bivme.fitting.perform_fit import perform_fitting
from bivme import TEST_RESOURCE_DIR, MODEL_RESOURCE_DIR
from . import CURRENT_RESIDUALS
import shutil


def test_performed_fit():
    test_data = ["patient_1_gpdata", "patient_2_gpdata"]
    output_dir = TEST_RESOURCE_DIR / 'output'

    for test_case in test_data:
        patient_name = test_case

        gp_file = TEST_RESOURCE_DIR / patient_name

        if not output_dir.exists():
            output_dir.mkdir()
        residuals = perform_fitting(gp_file, output_dir)

        assert residuals > 0
        assert round(residuals, 2) <= CURRENT_RESIDUALS[test_case]
        ##TODO update models and CURRENT_RESIDUALS for next tests - also add more cases
    shutil.rmtree(output_dir)


