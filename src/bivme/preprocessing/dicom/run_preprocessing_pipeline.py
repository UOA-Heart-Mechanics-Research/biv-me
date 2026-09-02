import os
from pathlib import Path
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
from bivme.preprocessing.dicom.export_guidepoints import export_guidepoints, apply_guidepoint_postprocessing
from bivme.plotting.plot_guidepoints import generate_html # for plotting guidepoints


def perform_preprocessing(case, config, mylogger):
    # Path: src/bivme/preprocessing/dicom/models
    MODEL_DIR = Path(os.path.dirname(__file__)) / 'models'

    # Unpack config parameters
    # Input
    start_time = time.time()
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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        mylogger.warning('MPS is available, but nnU-Net-v2 does not support it. It will use CPU instead, and this may be very slow!')
    else:
        device = torch.device("cpu")
        mylogger.warning('No GPU available. Using CPU instead. This may be very slow!')

    mylogger.info(f'Using device: {device}')

    ## Step 0: Pre-preprocessing (separate cines from non-cines)
    mylogger.info(f'Finding cines...')
    extract_cines(src, dst, mylogger)

    src = os.path.join(dst, 'processed-dicoms') # Update source directory
    mylogger.success(f'Pre-preprocessing complete. Cines extracted to {src}.')

    ## Step 1: View selection
    slice_info_df, num_phases = select_views(case, src, dst, MODEL_DIR, states, config, mylogger)

    if slice_info_df is None:
        mylogger.error('Skipping the rest of the preprocessing pipeline due to view selection failure.')
        return

    mylogger.success(f'View selection complete.')
    mylogger.info(f'Number of phases: {num_phases}')

    # Check if there's any 4ch selected
    if slice_info_df.empty:
        mylogger.error(f'No views were selected for case {case}. Please check the view selection output and adjust the view selection parameters in the config file if necessary.')
        return
    else:
        four_ch_views = slice_info_df[slice_info_df['View'] == '4ch']
        if four_ch_views.empty:
            mylogger.warning(f'No 4ch views were selected for case {case}. Segmentations and guidepoints will be created, but no meshes will be generated due to the lack of tricuspid valve points...')

    # # Step 2: Segmentation
    seg_start_time = time.time()
    mylogger.info(f'Starting segmentation...')
    segment_views(dst, MODEL_DIR, slice_info_df, config["multiprocessing"]["workers"], mylogger) # TODO: Find a way to suppress nnUnet output
    seg_end_time = time.time()
    mylogger.success(f'Segmentation complete. Time taken: {seg_end_time-seg_start_time} seconds.')

    ## Step 2.1: Correct phase mismatch (if required)
    correct_phase_mismatch(dst, slice_info_df, num_phases, mylogger) 

    ## Step 3: Guide point extraction
    slice_dict = generate_contours(dst, slice_info_df, num_phases, mylogger)
    mylogger.success(f'Guide points generated successfully.')

    ## Step 4: Export guide points
    export_guidepoints(dst, output, slice_dict, config["contouring"]["smooth_landmarks"])
    mylogger.success(f'Guide points exported successfully.')

    # Step 4.1: Post-process guidepoints (if desired)
    if config["contouring"]["apply_postprocessing"]:
        apply_guidepoint_postprocessing(output, case, mylogger)
        mylogger.success(f'Post-processing of guidepoints was successful.')

    mylogger.info(f'Total time taken for preprocessing: {time.time() - start_time} seconds.')

    if config["logging"]["generate_log_file"]:
        mylogger.remove(logger_id)
        # Copy log file to states directory
        shutil.copyfile(f'{output}/log_file_{time_string}.log', os.path.join(states, f'log_file_{time_string}.log'))