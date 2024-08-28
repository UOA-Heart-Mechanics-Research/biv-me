import numpy as np
from bivme.fitting.perform_fit import perform_fitting
from bivme import TEST_RESOURCE_DIR, MODEL_RESOURCE_DIR
from . import CURRENT_RESIDUALS
import shutil
from pathlib import Path
import tomli
def test_performed_fit():

    test_data = ["patient_1_gpdata", "patient_2_gpdata"]
    output_dir = TEST_RESOURCE_DIR / 'output'

    config_file = '../biv-me/src/bivme/configs/config.toml'
    with open(config_file, mode="rb") as fp:
        config = tomli.load(fp)

    # TOML Schema Validation
    match config:
        case {
            "input": {"gp_directory": str(),
                      "gp_suffix": str(),
                      "si_suffix": str(),
                      },
            "breathhold_correction": {"shifting": str(), "ed_frame": int()},
            "gp_processing": {"sampling": int(), "num_of_phantom_points_av": int(), "num_of_phantom_points_mv": int(), "num_of_phantom_points_tv": int(), "num_of_phantom_points_pv": int()},
            "multiprocessing": {"workers": int()},
            "fitting_weights": {"guide_points": float(), "convex_problem": float(), "transmural": float()},
            "output": {"output_directory": str(), "output_meshes": list(), "closed_mesh": bool(),  "show_logging": bool(), "export_control_mesh": bool(), "mesh_format": str(), "generate_log_file": bool(), "overwrite": bool()},
        }:
            pass
        case _:
            raise ValueError(f"Invalid configuration: {config}")

    for test_case in test_data:
        patient_name = test_case

        gp_file = TEST_RESOURCE_DIR / patient_name

        if not output_dir.exists():
            output_dir.mkdir()
        residuals = perform_fitting(gp_file, config, output_dir)

        assert residuals > 0
        assert round(residuals, 2) <= CURRENT_RESIDUALS[test_case]
        ##TODO update models and CURRENT_RESIDUALS for next tests - also add more cases
    shutil.rmtree(output_dir)


