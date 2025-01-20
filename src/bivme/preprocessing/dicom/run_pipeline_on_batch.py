import os,sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import shutil
import time

import warnings
warnings.filterwarnings('ignore')

# Import modules
from bivme.preprocessing.dicom.select_views import select_views
from bivme.preprocessing.dicom.segment_views import segment_views
from bivme.preprocessing.dicom.generate_contours import generate_contours
from bivme.preprocessing.dicom.export_guidepoints import export_guidepoints

from pathlib import Path
# Path: src/bivme/preprocessing/dicom/models
MODEL_DIR = Path(os.path.dirname(__file__)) / 'models'

if __name__ == "__main__":
    # Check if GPU is available (torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    batch_ID = 'unit-test'
    analyst_id = 'jdil469'

    src = r"C:\Users\jdil469\bivme-data\fitting\raw-dicoms" + "\\" + 'unit-test'
    dst = r"C:\Users\jdil469\bivme-data\fitting\processed" + "\\" + batch_ID
    states = r"C:\Users\jdil469\bivme-data\fitting\states" + "\\" + batch_ID
    output = r"C:\Users\jdil469\bivme-data\fitting\output" + "\\" + batch_ID
    model = MODEL_DIR

    caselist = os.listdir(src)

    os.makedirs(dst, exist_ok=True)
    os.makedirs(states, exist_ok=True)
    os.makedirs(output, exist_ok=True)

    for case in caselist:
        case_src = os.path.join(src, case)
        case_dst = os.path.join(dst, case)

        if not os.path.isdir(case_src):
            print(f'Processing case: {case}')
        if os.path.exists(case_dst):
            enter = input(f'Case {case} already processed. Do you want to overwrite? (y/n): ')
            if enter == 'y':
                shutil.rmtree(case_dst)
            else:
                print(f'Case {case} skipped.')
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

        print(f'Processing case: {case}')

        # Step 1: View selection
        option = 'default'
        slice_info_df, num_phases, slice_mapping = select_views(case, case_src, case_dst, model, states, option)

        # option = 'load'

        print('View selection complete.')

        print(f'Number of phases: {num_phases}')

        # Step 2: Segmentation
        ## Segmentation
        version = '3d' # 2d or 3d
        start_time = time.time()
        segment_views(case, dst, model, slice_info_df, version = version)
        end_time = time.time()
        print(f'Segmentation complete. Time taken: {end_time-start_time} seconds ({version} version).')

        # Add segmentation time to log
        with open(log_path, 'a') as f:
            f.write(f'Segmentation time: {end_time-start_time} seconds ({version} version).\n')

        # Step 3: Guide point extraction
        slice_dict = generate_contours(case, dst, slice_info_df, num_phases, version = version)
        print('Guide points generated.')

        # Step 4: Export guide points
        export_guidepoints(case, dst, output, slice_dict, slice_mapping)
        print('Export complete.')
        print(f'Case {case} complete.')
        print(f'Total time taken: {time.time()-start_time} seconds.')

