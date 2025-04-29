import os, sys
from pathlib import Path
import shutil
import argparse
import tomli
import datetime
from loguru import logger

from bivme.preprocessing.dicom.run_preprocessing_pipeline import perform_preprocessing
from bivme.preprocessing.dicom.run_preprocessing_pipeline import validate_config as validate_config_preprocessing
from bivme.fitting.perform_fit import perform_fitting
from bivme.fitting.perform_fit import validate_config as validate_config_fitting

def run_preprocessing(case, config, mylogger):
    try:
        perform_preprocessing(case, config, mylogger)
    except KeyboardInterrupt:
        mylogger.info(f"Program interrupted by the user")
        sys.exit(0)

def run_fitting(case, config, mylogger):
    try:
        if not config["logging"]["show_detailed_logging"]:
            mylogger.remove()

        mylogger.info(f"Processing {os.path.basename(case)}")
        if config["logging"]["generate_log_file"]:
            log_level = "DEBUG"
            log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
            logger_id = mylogger.add(f'{config["output_fitting"]["output_directory"]}/{os.path.basename(case)}/log_file_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log', level=log_level, format=log_format,
                                        colorize=False, backtrace=True,
                                        diagnose=True)
            
        folder = os.path.join(config["input_fitting"]["gp_directory"], case)
        residuals = perform_fitting(folder, config, out_dir=config["output_fitting"]["output_directory"], gp_suffix=config["input_fitting"]["gp_suffix"], si_suffix=config["input_fitting"]["si_suffix"],
                        frames_to_fit=[], output_format=config["output_fitting"]["mesh_format"], logger=logger)
        
        mylogger.info(f"Average residuals: {residuals} for case {os.path.basename(case)}")

        if config["logging"]["generate_log_file"]:
            mylogger.remove(logger_id)

    except KeyboardInterrupt:
        mylogger.info(f"Program interrupted by the user")
        sys.exit(0)

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

            "logging": {"show_detailed_logging": bool(), "generate_log_file": bool()},

            "input_pp": {"source": str(),
                      "batch_ID": str(),
                      "analyst_id": str(),
                      "processing": str(),
                      "states": str()
                      },
            "view-selection": {"option": str()},
            "segmentation": {"version": str()},
            "output_pp": {"overwrite": bool(), "generate_plots": bool(), "output_directory": str()},

            "input_fitting": {"gp_directory": str(),
                      "gp_suffix": str(),
                      "si_suffix": str(),
                      },
            "breathhold_correction": {"shifting": str(), "ed_frame": int()},
            "gp_processing": {"sampling": int(), "num_of_phantom_points_av": int(), "num_of_phantom_points_mv": int(), "num_of_phantom_points_tv": int(), "num_of_phantom_points_pv": int()},
            "multiprocessing": {"workers": int()},
            "fitting_weights": {"guide_points": float(), "convex_problem": float(), "transmural": float()},
            "output_fitting": {"output_directory": str(), "output_meshes": list(), "closed_mesh": bool(),   "export_control_mesh": bool(), "mesh_format": str(),  "overwrite": bool()},
        }:
            pass
        case _:
            raise ValueError(f"Invalid configuration: {config}")

    # Which modules are to be run?
    run_preprocessing_bool = config["modules"]["preprocessing"]
    run_fitting_bool = config["modules"]["fitting"]

    logger.info(f'Running modules: preprocessing={run_preprocessing_bool}, fitting={run_fitting_bool}')

    # Validate logging parameters
    if not (config["logging"]["show_detailed_logging"] == True or config["logging"]["show_detailed_logging"] == False):
        logger.error(f'Invalid logging option: {config["logging"]["show_detailed_logging"]}. Must be True or False.')
        sys.exit(0)
    if not (config["logging"]["generate_log_file"] == True or config["logging"]["generate_log_file"] == False):
        logger.error(f'Invalid logging option: {config["logging"]["generate_log_file"]}. Must be True or False.')
        sys.exit(0)
    
    # Determine which cases to process
    if run_preprocessing_bool: # then get case list from preprocessing source directory
        validate_config_preprocessing(config, logger)
        caselist = os.listdir(config["input_pp"]["source"])
        caselist = [case for case in caselist if os.path.isdir(os.path.join(config["input_pp"]["source"], case))]

        # Edit gp_directory to point to the output of the preprocessing
        gp_dir = os.path.join(config["output_pp"]["output_directory"], config["input_pp"]["batch_ID"])
        config["input_fitting"]["gp_directory"] = gp_dir

        # save a copy of the config file to the output folder
        output_folder = Path(config["output_pp"]["output_directory"])
        output_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config_file, output_folder)

    if run_fitting_bool: 
        validate_config_fitting(config, logger)

        if not run_preprocessing_bool: # then get case list from fitting source directory
            caselist = os.listdir(config["input_fitting"]["gp_directory"])
            caselist = [case for case in caselist if os.path.isdir(os.path.join(config["input_fitting"]["gp_directory"], case))]

        # save a copy of the config file to the output folder
        output_folder = Path(config["output_fitting"]["output_directory"])
        output_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config_file, output_folder)

    logger.info(f"Found {len(caselist)} cases to process.")

    for i, case in enumerate(caselist):
        logger.info(f"Processing case {i+1}/{len(caselist)}: {case}")

        if run_preprocessing_bool:
            if not config["output_pp"]["overwrite"] and os.path.exists(os.path.join(config["output_pp"]["output_directory"], config["input_pp"]["batch_ID"], case)):
                logger.info(f"Skipping preprocessing for {case} as it is already complete.")
                continue
            else:
                logger.info("Running preprocessing...")
                run_preprocessing(case, config, logger)
                logger.success("Preprocessing complete.")

        if run_fitting_bool:
            if not config["output_fitting"]["overwrite"] and os.path.exists(os.path.join(config["output_fitting"]["output_directory"], case)):
                logger.info(f"Skipping fitting for {case} as it is already complete.")
                continue
            else:
                logger.info("Running fitting...")
                run_fitting(case, config, logger)
                logger.success("Fitting complete.")




