import os
from pathlib import Path
import sys
import torch
import shutil
import time
import datetime
import argparse
import tomli
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


def run_preprocessing_pipeline(case, case_src, case_dst, model, states, option, version, output, plotting, my_logger):
    start_time = time.time()
    ## Step 0: Pre-preprocessing
    my_logger.info(f'Finding cines...')
    extract_cines(case_src, case_dst, my_logger)

    case_src = os.path.join(case_dst, 'processed-dicoms') # Update source directory
    my_logger.success(f'Pre-preprocessing complete. Cines extracted to {case_src}.')

    ## Step 1: View selection
    slice_info_df, num_phases, slice_mapping = select_views(case, case_src, case_dst, model, states, option, my_logger)

    my_logger.success(f'View selection complete.')
    my_logger.info(f'Number of phases: {num_phases}')

    ## Step 2: Segmentation
    seg_start_time = time.time()
    my_logger.info(f'Starting segmentation with {version} version...')
    segment_views(case, case_dst, model, slice_info_df, version, my_logger) # TODO: Find a way to suppress nnUnet output
    seg_end_time = time.time()
    my_logger.success(f'Segmentation complete. Time taken: {seg_end_time-seg_start_time} seconds ({version} version).')

    ## Step 2.1: Correct phase mismatch (if required)
    correct_phase_mismatch(case_dst, slice_info_df, num_phases, my_logger)

    ## Step 3: Guide point extraction
    slice_dict = generate_contours(case, case_dst, slice_info_df, num_phases, version, my_logger)
    my_logger.success(f'Guide points generated successfully.')

    ## Step 4: Export guide points
    export_guidepoints(case, case_dst, output, slice_dict, slice_mapping)
    my_logger.success(f'Guide points exported successfully.')
    my_logger.success(f'Case {case} complete.')
    my_logger.info(f'Total time taken: {time.time()-start_time} seconds.')

    ## Step 5: Generate HTML (optional) of guide points for visualisation
    gp_dir = os.path.join(output, case)
    generate_html(gp_dir, out_dir=plotting, gp_suffix='', si_suffix='', frames_to_fit=[], my_logger=my_logger, model_path=None)

    logger.info(f'Guidepoints plotted and saved in {plotting}.')

def preprocess_cases(config, mylogger):
    # Path: src/bivme/preprocessing/dicom/models
    MODEL_DIR = Path(os.path.dirname(__file__)) / 'models'

    # Unpack config
    src = config["input_pp"]["source"]

    assert os.path.exists(src), \
        f'DICOM folder does not exist! Make sure to add the correct directory under "source" in the config file.'

    batch_ID = config["input_pp"]["batch_ID"]
    analyst_id = config["input_pp"]["analyst_id"]

    dst = os.path.join(config["input_pp"]["processing"], batch_ID)
    os.makedirs(dst, exist_ok=True)

    states = os.path.join(config["input_pp"]["states"], batch_ID)
    os.makedirs(states, exist_ok=True)

    overwrite = config["output_pp"]["overwrite"]

    output = os.path.join(config["output_pp"]["output_directory"], batch_ID)
    os.makedirs(output, exist_ok=True)

    plotting = dst # save the plotted htmls in processed directory

    option = config["view-selection"]["option"]
    version = config["segmentation"]["version"]

    # Define list of cases to process
    caselist = os.listdir(src)

    mylogger.info(f'{len(caselist)} case(s) found.')

    # Set up logging
    log_level = "DEBUG"
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"

    # Check if GPU is available (torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mylogger.info(f'Using device: {device}')

    try:
        for i, case in enumerate(caselist):
            mylogger.info(f'Processing case: {i+1}/{len(caselist)}')
            case_src = os.path.join(src, case)
            case_dst = os.path.join(dst, case)
            case_states = os.path.join(states, case, analyst_id)
            os.makedirs(case_states, exist_ok=True)

            logger_id = mylogger.add(f'{case_states}/log_file_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log', level=log_level, format=log_format,
                colorize=False, backtrace=True,
                diagnose=True)

            if os.path.exists(case_dst):
                if overwrite:
                    mylogger.info(f'Overwriting case: {case}')
                    shutil.rmtree(case_dst)
                else:
                    mylogger.info(f'Skipping already processed case: {case}')
                    continue
                    
            mylogger.info(f'Processing case: {case}')

            run_preprocessing_pipeline(case, case_src, case_dst, MODEL_DIR, case_states, option, version, output, plotting, mylogger)

            mylogger.remove(logger_id)

    except KeyboardInterrupt:
        mylogger.info(f"Program interrupted by the user")
        sys.exit(0)

