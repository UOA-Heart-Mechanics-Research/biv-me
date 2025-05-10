import os
from pathlib import Path
import sys
import torch
import time
import shutil
import datetime
from loguru import logger

import warnings
warnings.filterwarnings('ignore')

# Import modules
from bivme.preprocessing.dicom.extract_cines import extract_cines
from bivme.preprocessing.dicom.select_views import select_views
from bivme.preprocessing.dicom.segment_views import segment_views
from bivme.preprocessing.dicom.correct_phase_mismatch import correct_phase_mismatch
from bivme.preprocessing.dicom.generate_contours import generate_contours
from bivme.preprocessing.dicom.export_guidepoints import export_guidepoints
from bivme.plotting.plot_guidepoints import generate_html # for plotting guidepoints


def perform_preprocessing(case, config, mylogger):
    # Path: src/bivme/preprocessing/dicom/models
    MODEL_DIR = Path(os.path.dirname(__file__)) / 'models'

    # Unpack config parameters
    # Input
    src = os.path.join(config["input_pp"]["source"], case)

    # Processing
    dst = os.path.join(config["input_pp"]["processing"], config["input_pp"]["batch_ID"])
    dst = os.path.join(dst, case) # destination directory for processed files
    if os.path.exists(dst):
        shutil.rmtree(dst) # remove existing directory
    os.makedirs(dst, exist_ok=True) # create new directory
    
    states = os.path.join(config["input_pp"]["states"], config["input_pp"]["batch_ID"])
    states = os.path.join(states, case, config["input_pp"]["analyst_id"]) # destination directory for view predictions which don't get overwritten, and log files
    os.makedirs(states, exist_ok=True)

    # Output
    output = os.path.join(config["output_pp"]["output_directory"], config["input_pp"]["batch_ID"])
    output = os.path.join(output, case) # output directory for guidepoints
    if os.path.exists(output):
        shutil.rmtree(output) # remove existing directory
    os.makedirs(output, exist_ok=True) # create new directory

    plotting = dst # save the plotted htmls in processed directory

    # Logging
    if not config["logging"]["show_detailed_logging"]:
        mylogger.remove()

    if config["logging"]["generate_log_file"]:
        log_level = "DEBUG"
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
        time_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        logger_id = mylogger.add(f'{output}/log_file_{time_string}.log', level=log_level, format=log_format,
                    colorize=False, backtrace=True,
                    diagnose=True)

    # Check if GPU is available (torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mylogger.info(f'Using device: {device}')

    ## Step 0: Pre-preprocessing (separate cines from non-cines)
    mylogger.info(f'Finding cines...')
    extract_cines(src, dst, mylogger)

    src = os.path.join(dst, 'processed-dicoms') # Update source directory
    mylogger.success(f'Pre-preprocessing complete. Cines extracted to {src}.')

    ## Step 1: View selection
    slice_info_df, num_phases, slice_mapping = select_views(case, src, dst, MODEL_DIR, states, config["view-selection"]["option"], mylogger)

    mylogger.success(f'View selection complete.')
    mylogger.info(f'Number of phases: {num_phases}')

    ## Step 2: Segmentation
    seg_start_time = time.time()
    mylogger.info(f'Starting segmentation with {config["segmentation"]["version"]} version...')
    segment_views(dst, MODEL_DIR, slice_info_df, config["segmentation"]["version"], mylogger) # TODO: Find a way to suppress nnUnet output
    seg_end_time = time.time()
    mylogger.success(f'Segmentation complete. Time taken: {seg_end_time-seg_start_time} seconds ({config["segmentation"]["version"]} version).')

    ## Step 2.1: Correct phase mismatch (if required)
    correct_phase_mismatch(dst, slice_info_df, num_phases, mylogger)

    ## Step 3: Guide point extraction
    slice_dict = generate_contours(dst, slice_info_df, num_phases, config["segmentation"]["version"], mylogger)
    mylogger.success(f'Guide points generated successfully.')

    ## Step 4: Export guide points
    export_guidepoints(dst, output, slice_dict, slice_mapping)
    mylogger.success(f'Guide points exported successfully.')

    ## Step 5: Generate HTML (optional) of guide points for visualisation
    if config["output_pp"]["generate_plots"]:
        generate_html(output, out_dir=plotting, gp_suffix='', si_suffix='', frames_to_fit=[], my_logger=mylogger, model_path=None)

    mylogger.success(f'Guidepoints plotted and saved in {plotting}.')

    if config["logging"]["generate_log_file"]:
        mylogger.remove(logger_id)
        # Copy log file to states directory
        shutil.copyfile(f'{output}/log_file_{time_string}.log', os.path.join(states, f'log_file_{time_string}.log'))

def validate_config(config, mylogger):
    assert os.path.exists(config["input_pp"]["source"]), \
        f'DICOM folder does not exist! Make sure to add the correct directory under "source" in the config file.'
    
    if not (config["view-selection"]["option"] == "default" or config["view-selection"]["option"] == "metadata-only" 
            or config["view-selection"]["option"] == "image-only"  or config["view-selection"]["option"] == "load"):
        mylogger.error(f'Invalid view selection option: {config["view-selection"]["option"]}. Must be "default", "metadata-only", "image-only", or "load".')
        sys.exit(0)

    if not (config["segmentation"]["version"] == "2d" or config["segmentation"]["version"] == "3d"):
        mylogger.error(f'Invalid segmentation version: {config["segmentation"]["version"]}. Must be "2d" or "3d".')
        sys.exit(0)

    if not (config["output_pp"]["overwrite"] == True or config["output_pp"]["overwrite"] == False):
        mylogger.error(f'Invalid overwrite option: {config["output_pp"]["overwrite"]}. Must be true or false.')
        sys.exit(0)

    if not (config["output_pp"]["generate_plots"] == True or config["output_pp"]["generate_plots"] == False):
        mylogger.error(f'Invalid generate_plots option: {config["output_pp"]["generate_plots"]}. Must be true or false.')
        sys.exit(0)
