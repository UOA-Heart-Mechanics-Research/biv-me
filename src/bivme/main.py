import os, sys
from pathlib import Path
import shutil
import argparse
import tomli
import datetime
import time
from loguru import logger

from bivme.preprocessing.dicom.run_preprocessing_pipeline import perform_preprocessing
from bivme.fitting.perform_fit import perform_fitting
from bivme.plotting.plot_guidepoints import generate_html


def validate_config_preprocessing(config, mylogger):
    assert os.path.exists(config["input_pp"]["source"]), \
        f'DICOM folder does not exist! Make sure to add the correct directory under "source" in the config file.'
    
    if not (config["view-selection"]["option"] == "default" or config["view-selection"]["option"] == "image-only"  or config["view-selection"]["option"] == "load"):
        mylogger.error(f'Invalid view selection option: {config["view-selection"]["option"]}. Must be "default", "image-only", or "load".')
        sys.exit(0)

    if not (config["view-selection"]["correct_mode"] == "automatic" or config["view-selection"]["correct_mode"] == "adaptive" or config["view-selection"]["correct_mode"] == "manual"):
        mylogger.error(f'Invalid correct mode: {config["view-selection"]["correct_mode"]}. Must be "automatic", "adaptive", or "manual".')
        sys.exit(0)

    if not (config["view-selection"]["show_videos"] == True or config["view-selection"]["show_videos"] == False):
        mylogger.error(f'Invalid show_videos option: {config["view-selection"]["show_videos"]}. Must be true or false.')
        sys.exit(0)

    if not (config["contouring"]["smooth_landmarks"] == True or config["contouring"]["smooth_landmarks"] == False):
        mylogger.error(f'Invalid smooth_landmarks option: {config["contouring"]["smooth_landmarks"]}. Must be true or false.')
        sys.exit(0)

    if not (config["contouring"]["apply_postprocessing"] == True or config["contouring"]["apply_postprocessing"] == False):
        mylogger.error(f'Invalid apply_postprocessing option: {config["contouring"]["apply_postprocessing"]}. Must be true or false.')
        sys.exit(0)

    if not (config["output_pp"]["overwrite"] == True or config["output_pp"]["overwrite"] == False):
        mylogger.error(f'Invalid overwrite option: {config["output_pp"]["overwrite"]}. Must be true or false.')
        sys.exit(0)


def validate_config_fitting(config, mylogger):
    assert Path(config["input_fitting"]["gp_directory"]).exists(), \
        f'gp_directory does not exist. Cannot find {config["input_fitting"]["gp_directory"]}!'

    if not (config["breathhold_correction"]["shifting"] == "derived_from_ed" or config["breathhold_correction"]["shifting"] == "average_all_frames" or config["breathhold_correction"]["shifting"] == "none"):
        mylogger.error(f'argument shifting must be derived_from_ed, average_all_frames or none. {config["breathhold_correction"]["shifting"]} given.')
        sys.exit(0)

    if not config["breathhold_correction"]["ed_frame"] >= 0:
        mylogger.error(f'argument ed_frame must be a non-negative integer. {config["breathhold_correction"]["ed_frame"]} given.')
        sys.exit(0)

    if not config["gp_processing"]["sampling"] > 0:
        mylogger.error(f'argument sampling must be a positive integer. {config["gp_processing"]["sampling"]} given.')
        sys.exit(0)
    
    if not config["gp_processing"]["num_of_phantom_points_av"] >= 0:
        mylogger.error(f'argument num_of_phantom_points_av must be a non-negative integer. {config["gp_processing"]["num_of_phantom_points_av"]} given.')
        sys.exit(0)

    if not config["gp_processing"]["num_of_phantom_points_mv"] >= 0:
        mylogger.error(f'argument num_of_phantom_points_mv must be a non-negative integer. {config["gp_processing"]["num_of_phantom_points_mv"]} given.')
        sys.exit(0)

    if not config["gp_processing"]["num_of_phantom_points_tv"] >= 0:
        mylogger.error(f'argument num_of_phantom_points_tv must be a non-negative integer. {config["gp_processing"]["num_of_phantom_points_tv"]} given.')
        sys.exit(0)

    if not config["gp_processing"]["num_of_phantom_points_pv"] >= 0:
        mylogger.error(f'argument num_of_phantom_points_pv must be a non-negative integer. {config["gp_processing"]["num_of_phantom_points_pv"]} given.')
        sys.exit(0)

    if not config["fitting_weights"]["guide_points"] >= 0:
        mylogger.error(f'argument guide_points weight must be a non-negative number. {config["fitting_weights"]["guide_points"]} given.')
        sys.exit(0)

    if not config["fitting_weights"]["convex_problem"] >= 0:
        mylogger.error(f'argument convex_problem weight must be a non-negative number. {config["fitting_weights"]["convex_problem"]} given.')
        sys.exit(0)

    if not config["fitting_weights"]["transmural"] >= 0:
        mylogger.error(f'argument transmural weight must be a non-negative number. {config["fitting_weights"]["transmural"]} given.')
        sys.exit(0)

    if not config["fitting_weights"]["lsq_trans_weight"] >= 0:
        mylogger.error(f'argument lsq_trans_weight must be a non-negative number. {config["fitting_weights"]["lsq_trans_weight"]} given.')
        sys.exit(0)

    if not (config["refitting"]["perform_refits"] == True or config["refitting"]["perform_refits"] == False):
        mylogger.error(f'argument perform_refits must be true or false. {config["refitting"]["perform_refits"]} given.')
        sys.exit(0)

    if not config["refitting"]["threshold_factor"] >= 1:
        mylogger.error(f'argument threshold_factor must be a number greater than or equal to 1. {config["refitting"]["threshold_factor"]} given.')
        sys.exit(0)

    if not (config["output_fitting"]["mesh_format"].endswith('.obj') or config["output_fitting"]["mesh_format"].endswith('.vtk') or config["output_fitting"]["mesh_format"] == 'none'):
        mylogger.error(f'argument mesh_format must be .obj, .vtk or none. {config["output_fitting"]["mesh_format"]} given.')
        sys.exit(0)

    for mesh in config["output_fitting"]["output_meshes"]:
        if mesh not in ["LV_ENDOCARDIAL", "RV_ENDOCARDIAL", "EPICARDIAL"]:
            mylogger.error(f'argument output_meshes invalid. {mesh} given. Allowed values are "LV_ENDOCARDIAL", "RV_ENDOCARDIAL", "EPICARDIAL"')
            sys.exit(0)

    if not (config["output_fitting"]["mesh_format"].endswith('.obj') or config["output_fitting"]["mesh_format"].endswith('.vtk') or config["output_fitting"]["mesh_format"] == 'none'):
        mylogger.error(f'argument mesh_format must be .obj, .vtk or none. {config["output_fitting"]["mesh_format"]} given.')
        sys.exit(0)

    if not (config["output_fitting"]["closed_mesh"] == True or config["output_fitting"]["closed_mesh"] == False):
        mylogger.error(f'argument closed_mesh must be true or false. {config["output_fitting"]["closed_mesh"]} given.')
        sys.exit(0)

    if not (config["output_fitting"]["overwrite"] == True or config["output_fitting"]["overwrite"] == False):
        mylogger.error(f'argument overwrite must be true or false. {config["output_fitting"]["overwrite"]} given.')
        sys.exit(0)

    if not config["multiprocessing"]["workers"] > 0:
        mylogger.error(f'argument workers must be a positive integer. {config["multiprocessing"]["workers"]} given.')
        sys.exit(0)

def validate_config_misc(config, mylogger):
    # Validate logging parameters
    if not (config["logging"]["show_detailed_logging"] == True or config["logging"]["show_detailed_logging"] == False):
        mylogger.error(f'Invalid logging option: {config["logging"]["show_detailed_logging"]}. Must be True or False.')
        sys.exit(0)
    if not (config["logging"]["generate_log_file"] == True or config["logging"]["generate_log_file"] == False):
        mylogger.error(f'Invalid logging option: {config["logging"]["generate_log_file"]}. Must be True or False.')
        sys.exit(0)

    # Validate plotting parameters
    if not (config["plotting"]["generate_plots_preprocessing"] == True or config["plotting"]["generate_plots_preprocessing"] == False):
        mylogger.error(f'Invalid plotting option: {config["plotting"]["generate_plots_preprocessing"]}. Must be True or False.')
        sys.exit(0)
    if not (config["plotting"]["generate_plots_fitting"] == True or config["plotting"]["generate_plots_fitting"] == False):
        mylogger.error(f'Invalid plotting option: {config["plotting"]["generate_plots_fitting"]}. Must be True or False.')
        sys.exit(0)
    if not (config["plotting"]["include_images"] == True or config["plotting"]["include_images"] == False):
        mylogger.error(f'Invalid plotting option: {config["plotting"]["include_images"]}. Must be True or False.')
        sys.exit(0)
    if not (config["plotting"]["export_images"] == True or config["plotting"]["export_images"] == False):
        mylogger.error(f'Invalid plotting option: {config["plotting"]["export_images"]}. Must be True or False.')
        sys.exit(0)
    

def run_preprocessing(case, config, mylogger):
    try:
        perform_preprocessing(case, config, mylogger)

        if config["plotting"]["generate_plots_preprocessing"]:
            start_time = time.time()
            output = os.path.join(config["output_pp"]["output_directory"], config["input_pp"]["batch_ID"], case)  # output directory for guidepoints
            plotting = os.path.join(config["input_pp"]["processing"], config["input_pp"]["batch_ID"]) # save the plotted htmls in processed directory

            if config["plotting"]["include_images"]:
                dst = os.path.join(config["input_pp"]["processing"], config["input_pp"]["batch_ID"], case)  # processed directory
                image_path = os.path.join(dst, 'images')  # Path to the image file for plotting
            else:
                image_path = None

            generate_html(case, output, out_dir=plotting, gp_suffix='', si_suffix='', frames_to_fit=[], my_logger=mylogger, model_path=None, image_path=image_path, vtk_export_path=None, workers=config["multiprocessing"]["workers"])

            mylogger.info(f'Generated html plots (preprocessing) at {os.path.join(plotting,case,"html")}.')
            mylogger.info(f"Generating plots took: {time.time() - start_time:.2f} seconds")

    except KeyboardInterrupt:
        mylogger.info(f"Program interrupted by the user")
        sys.exit(0)


def run_fitting(case, config, mylogger):
    try:
        if not config["logging"]["show_detailed_logging"]:
            mylogger.remove()

        mylogger.info(f"Processing {os.path.basename(case)}")

        # Remove existing models if overwrite is true
        if os.path.exists(os.path.join(config["output_fitting"]["output_directory"], case)) and config["output_fitting"]["overwrite"]: 
            shutil.rmtree(os.path.join(config["output_fitting"]["output_directory"], case), ignore_errors=True)

        if config["logging"]["generate_log_file"]:
            log_level = "DEBUG"
            log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
            logger_id = mylogger.add(f'{config["output_fitting"]["output_directory"]}/{os.path.basename(case)}/log_file_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log', level=log_level, format=log_format,
                                        colorize=False, backtrace=True,
                                        diagnose=True)
            
        folder = os.path.join(config["input_fitting"]["gp_directory"], case)
        gp_suffix = config["input_fitting"]["gp_suffix"]
        si_suffix = config["input_fitting"]["si_suffix"]

        start_time = time.time()
        residuals = perform_fitting(folder, config, out_dir=config["output_fitting"]["output_directory"], gp_suffix=gp_suffix, si_suffix=si_suffix,
                        workers=config["multiprocessing"]["workers"], output_format=config["output_fitting"]["mesh_format"], my_logger=logger)
        mylogger.info(f"Fitting models for case {os.path.basename(case)} took: {time.time() - start_time:.2f} seconds")
        mylogger.info(f"Average residuals: {residuals} for case {os.path.basename(case)}")

        if config["plotting"]["generate_plots_fitting"] and residuals != -1:
            start_time = time.time()
            gp_dir = os.path.join(config["output_fitting"]["output_directory"], case, "gpfiles") # directory where the guidepoints are saved after fitting
            model_dir = os.path.join(config["output_fitting"]["output_directory"], case) # directory where the fitted models are saved
            out_dir = os.path.join(config["output_fitting"]["output_directory"]) # output directory for the html plot\

            if config["plotting"]["include_images"]:
                image_path = os.path.join(config["input_pp"]["processing"], config["input_pp"]["batch_ID"], case, "images")
            else:
                image_path = None

            if config["plotting"]["export_images"]:
                vtk_export_path = os.path.join(config["output_fitting"]["output_directory"], case, "vtk-images")
                os.makedirs(vtk_export_path, exist_ok=True)
            else:
                vtk_export_path = None

            generate_html(case, gp_dir=gp_dir, out_dir=out_dir, gp_suffix=gp_suffix, si_suffix=si_suffix,
                frames_to_fit=[], my_logger=logger, model_path = model_dir, image_path = image_path, vtk_export_path=vtk_export_path, workers=config["multiprocessing"]["workers"])
            
            mylogger.info(f"Generated html plots (fitting) for case {case} at {os.path.join(out_dir,case,f'html{gp_suffix}')}.")
            mylogger.info(f"Generating plots took: {time.time() - start_time:.2f} seconds")
            
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
    parser.add_argument('-case', '--case_name', type=str, 
                        help='Case name to process (only this case will be run)', default=None)

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

            "plotting": {"generate_plots_preprocessing": bool(), "generate_plots_fitting": bool(), "include_images": bool(), "export_images": bool()},

            "input_pp": {"source": str(),
                        "batch_ID": str(),
                        "analyst_id": str(),
                        "processing": str(),
                        "states": str()
                        },
            "view-selection": {"option": str(), "correct_mode": str(), "show_videos": bool()},
            "contouring": {"smooth_landmarks": bool(), "apply_postprocessing": bool()},
            "output_pp": {"overwrite": bool(), "output_directory": str()},

            "input_fitting": {"gp_directory": str(),
                        "gp_suffix": str(),
                        "si_suffix": str(),
                        },
            "breathhold_correction": {"shifting": str(), "ed_frame": int()},
            "gp_processing": {"sampling": int(), "num_of_phantom_points_av": int(), "num_of_phantom_points_mv": int(), "num_of_phantom_points_tv": int(), "num_of_phantom_points_pv": int()},
            "multiprocessing": {"workers": int()},
            "fitting_weights": {"guide_points": float(), "convex_problem": float(), "transmural": float(), "lsq_trans_weight": float()},
            "refitting": {"perform_refits": bool(), "threshold_factor": float()},
            "output_fitting": {"output_directory": str(), "output_meshes": list(), "closed_mesh": bool(),   "export_control_mesh": bool(), "mesh_format": str(),  "overwrite": bool()},
        }:
            pass
        case _:
            raise ValueError(f"Invalid configuration: {config}")

    # Which modules are to be run?
    run_preprocessing_bool = config["modules"]["preprocessing"]
    run_fitting_bool = config["modules"]["fitting"]

    logger.info(f'Running modules: preprocessing={run_preprocessing_bool}, fitting={run_fitting_bool}')

    # Validate logging and plotting parameters
    validate_config_misc(config, logger)
    
    # Determine which cases to process
    if run_preprocessing_bool: # then get case list from preprocessing source directory
        validate_config_preprocessing(config, logger)
        caselist = os.listdir(config["input_pp"]["source"])
        caselist = [case for case in caselist if os.path.isdir(os.path.join(config["input_pp"]["source"], case))]

        if args.case_name is not None:
            if args.case_name in caselist:
                caselist = [args.case_name]
            else:
                logger.error(f'Case {args.case_name} not found in source directory {config["input_pp"]["source"]}.')
                sys.exit(0)

        # Edit gp_directory to point to the output of the preprocessing
        gp_dir = os.path.join(config["output_pp"]["output_directory"], config["input_pp"]["batch_ID"])
        config["input_fitting"]["gp_directory"] = gp_dir
            
        os.makedirs(gp_dir, exist_ok=True)
        shutil.copy(args.config_file, gp_dir)

    if run_fitting_bool: 
        validate_config_fitting(config, logger)

        if not run_preprocessing_bool: # then get case list from fitting source directory
            caselist = os.listdir(config["input_fitting"]["gp_directory"])
            caselist = [case for case in caselist if os.path.isdir(os.path.join(config["input_fitting"]["gp_directory"], case))]

        if args.case_name is not None:
            if args.case_name in caselist:
                caselist = [args.case_name]
            else:
                logger.error(f'Case {args.case_name} not found in gp_directory {config["input_fitting"]["gp_directory"]}.')
                sys.exit(0)

        # save a copy of the config file to the output folder
        output_folder = Path(config["output_fitting"]["output_directory"])
        output_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config_file, output_folder)

    logger.info(f"Found {len(caselist)} cases to process.")

    # Sort caselist
    caselist.sort()

    for i, case in enumerate(caselist):
        logger.info(f"Processing case {i+1}/{len(caselist)}: {case}")

        if run_preprocessing_bool:
            if not config["output_pp"]["overwrite"] and os.path.exists(os.path.join(config["output_pp"]["output_directory"], config["input_pp"]["batch_ID"], case)):
                logger.info(f"Skipping preprocessing for {case} as it is already complete at {os.path.join(config['output_pp']['output_directory'], config['input_pp']['batch_ID'], case)}.")
                continue
            else:
                logger.info("Running preprocessing...")
                run_preprocessing(case, config, logger)
                if not config["logging"]["show_detailed_logging"]:
                    logger.add(sys.stderr) # need add sink for log again
                logger.success("Preprocessing complete.")

        if run_fitting_bool:
            if not config["output_fitting"]["overwrite"] and os.path.exists(os.path.join(config["output_fitting"]["output_directory"], case)) and len(os.listdir(os.path.join(config["output_fitting"]["output_directory"], case))) > 0:
                logger.info(f"Skipping fitting for {case} as it is already complete at {os.path.join(config['output_fitting']['output_directory'], case)}.")
                continue
            else:
                logger.info("Running fitting...")
                run_fitting(case, config, logger)
                if not config["logging"]["show_detailed_logging"]:
                    logger.add(sys.stderr) # need add sink for log again
                logger.success("Fitting complete.")




