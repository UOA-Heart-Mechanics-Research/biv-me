import os
import torch
import shutil
import time
import argparse
import tomli

import warnings
warnings.filterwarnings('ignore')

# Import modules
from bivme.preprocessing.dicom.select_views import select_views
from bivme.preprocessing.dicom.segment_views import segment_views
from bivme.preprocessing.dicom.generate_contours import generate_contours
from bivme.preprocessing.dicom.export_guidepoints import export_guidepoints


def run_pipeline(case, case_src, case_dst, model, states, option, version, output, log_path):

    ## Step 1: View selection
    slice_info_df, num_phases, slice_mapping = select_views(case, case_src, case_dst, model, states, option)

    print('View selection complete.\n')

    print(f'Number of phases: {num_phases}\n')

    ## Step 2: Segmentation
    start_time = time.time()
    print(f'Starting segmentation with {version} version...\n')
    segment_views(case, case_dst, model, slice_info_df, version)
    end_time = time.time()
    print(f'Segmentation complete. Time taken: {end_time-start_time} seconds ({version} version).\n')

    # Add segmentation time to log
    with open(log_path, 'a') as f:
        f.write(f'Segmentation time: {end_time-start_time} seconds ({version} version).\n')

    ## Step 3: Guide point extraction
    slice_dict = generate_contours(case, case_dst, slice_info_df, num_phases, version)
    print('Guide points generated.\n')

    ## Step 4: Export guide points
    export_guidepoints(case, case_dst, output, slice_dict, slice_mapping)
    print('Export complete.\n')
    print(f'Case {case} complete.\n')
    print(f'Total time taken: {time.time()-start_time} seconds.\n')

if __name__ == "__main__":
    # Check if GPU is available (torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    parser = argparse.ArgumentParser(description='Preprocess DICOM files for fitting')
    parser.add_argument('-config', '--config_file', type=str,
                        help='Config file containing preprocessing parameters', default='configs/preprocess_config.toml')
    args = parser.parse_args()

    from pathlib import Path
    # Path: src/bivme/preprocessing/dicom/models
    MODEL_DIR = Path(os.path.dirname(__file__)) / 'models'

    # Load config
    assert Path(args.config_file).exists(), \
        f'Cannot not find {args.config_file}!'
    with open(args.config_file, mode="rb") as fp:
        print(f'Loading config file: {args.config_file}')
        config = tomli.load(fp)

    # TOML Schema Validation
    match config:
        case {
            "input": {"source": str(),
                      "batch_ID": str(),
                      "analyst_id": str(),
                      "processing": str(),
                      "states": str()
                      },
            "view classification": {"option": str()},
            "segmentation": {"version": str()},
            "output": {"output_directory": str(), "overwrite": bool()},
        }:
            pass
        case _:
            raise ValueError(f"Invalid configuration: {config}")

    # Unpack config
    src = config["input"]["source"]

    assert os.path.exists(src), \
        f'DICOM folder does not exist! Make sure to add the correct directory under "source" in the config file.'

    batch_ID = config["input"]["batch_ID"]
    analyst_id = config["input"]["analyst_id"]

    dst = os.path.join(config["input"]["processing"], batch_ID)
    os.makedirs(dst, exist_ok=True)

    states = os.path.join(config["input"]["states"], batch_ID)
    os.makedirs(states, exist_ok=True)

    overwrite = config["output"]["overwrite"]
    output = os.path.join(config["output"]["output_directory"], batch_ID)
    os.makedirs(output, exist_ok=True)

    option = config["view classification"]["option"]
    version = config["segmentation"]["version"]

    caselist = os.listdir(src)

    print(f'{len(caselist)} case(s) found.\n')

    for i, case in enumerate(caselist):
        print(f'Processing case: {i+1}/{len(caselist)}\n')
        case_src = os.path.join(src, case)
        case_dst = os.path.join(dst, case)

        if os.path.exists(case_dst):
            if overwrite:
                print(f'Overwriting already processed case: {case}\n')
                shutil.rmtree(case_dst)
            else:
                print(f'Skipping already processed case: {case}\n')
                continue
                
        start_time = time.time()
        # create log file to track manual inputs and other misc info
        case_states = os.path.join(states, case, analyst_id)
        os.makedirs(case_states, exist_ok=True)
        if not os.path.exists(os.path.join(case_states, 'logs')):
            os.mkdir(os.path.join(case_states, 'logs'))

        log_name = f'{case}_{analyst_id}_{time.ctime().replace(" ","-").replace(":","-")}.txt'
        log_path = os.path.join(case_states, 'logs', log_name)
        with open(log_path, 'w') as f:
            f.write(f'Case: {case}\n')
            f.write(f'Batch ID: {batch_ID}\n')
            f.write(f'Source: {case_src}\n')
            f.write(f'Destination: {case_dst}\n')
            f.write(f'Analyst ID: {analyst_id}\n')
            f.write(f'Processing started at: {time.ctime()}\n')
            f.write('\n')

        print(f'Processing case: {case}\n')

        run_pipeline(case, case_src, case_dst, MODEL_DIR, case_states, option, version, output, log_path)

