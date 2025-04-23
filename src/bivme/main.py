import os
from pathlib import Path
import shutil
import argparse
import tomli
from loguru import logger

from bivme.fitting.perform_fit import fit_cases
from bivme.preprocessing.dicom.run_preprocessing_pipeline import preprocess_cases

def run_preprocessing(config, mylogger):
    preprocess_cases(config, mylogger)

def run_fitting(config, mylogger):
    fit_cases(config, mylogger)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run biv-me modules')
    parser.add_argument('-config', '--config_file', type=str,
                        help='Config file describing which modules to run and their associated parameters', default='configs/config.toml')
    args = parser.parse_args()

    # Set up logging
    log_level = "DEBUG"
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"

    # Load config
    assert Path(args.config_file).exists(), \
        f'Cannot not find {args.config_file}!'
    with open(args.config_file, mode="rb") as fp:
        logger.info(f'Loading config file: {args.config_file}')
        config = tomli.load(fp)

    # TOML Schema Validation
    match config:
        case {
            "modules": {"preprocessing": bool(), "fitting": bool()},

            "input_pp": {"source": str(),
                      "batch_ID": str(),
                      "analyst_id": str(),
                      "processing": str(),
                      "states": str()
                      },
            "view-selection": {"option": str()},
            "segmentation": {"version": str()},
            "output_pp": {"output_directory": str(), "overwrite": bool()},

            "input_fitting": {"gp_directory": str(),
                      "gp_suffix": str(),
                      "si_suffix": str(),
                      },
            "breathhold_correction": {"shifting": str(), "ed_frame": int()},
            "gp_processing": {"sampling": int(), "num_of_phantom_points_av": int(), "num_of_phantom_points_mv": int(), "num_of_phantom_points_tv": int(), "num_of_phantom_points_pv": int()},
            "multiprocessing": {"workers": int()},
            "fitting_weights": {"guide_points": float(), "convex_problem": float(), "transmural": float()},
            "output_fitting": {"output_directory": str(), "output_meshes": list(), "closed_mesh": bool(),  "show_logging": bool(), "export_control_mesh": bool(), "mesh_format": str(), "generate_log_file": bool(), "overwrite": bool()},
        }:
            pass
        case _:
            raise ValueError(f"Invalid configuration: {config}")

    # Which modules are to be run?
    run_preprocessing_bool = config["modules"]["preprocessing"]
    run_fitting_bool = config["modules"]["fitting"]


    logger.info(f'Running modules: preprocessing={run_preprocessing_bool}, fitting={run_fitting_bool}')

    if run_preprocessing_bool:
        logger.info("Running preprocessing...")

        run_preprocessing(config, logger)
    
    if run_fitting_bool:
        logger.info("Running fitting...")

        if run_preprocessing_bool:
            # Edit gp_directory to point to the output of the preprocessing
            gp_dir = os.path.join(config["output_pp"]["output_directory"], config["input_pp"]["batch_ID"])
            config["input_fitting"]["gp_directory"] = gp_dir

        # save config file to the output folder
        output_folder = Path(config["output_fitting"]["output_directory"])
        output_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config_file, output_folder)

        run_fitting(config, logger)


