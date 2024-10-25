import os, sys
import numpy as np
import time
import plotly.graph_objs as go
from pathlib import Path
from plotly.offline import plot
import argparse
import pathlib
import datetime
import tomli
import shutil
import re
import fnmatch
from bivme.fitting.BiventricularModel import BiventricularModel
from bivme.fitting.GPDataSet import GPDataSet
from bivme.fitting.surface_enum import ContourType
from loguru import logger
from rich.progress import Progress
from bivme import MODEL_RESOURCE_DIR

# This list of contours_to _plot was taken from Liandong Lee
contours_to_plot = [
    ContourType.LAX_RA,
    ContourType.LAX_LA,
    ContourType.SAX_RA,
    ContourType.SAX_LA,
    ContourType.SAX_LAA,
    ContourType.SAX_LPV,
    ContourType.SAX_RPV,
    ContourType.SAX_SVC,
    ContourType.SAX_IVC,
    ContourType.LAX_RV_ENDOCARDIAL,
    ContourType.SAX_RV_FREEWALL,
    ContourType.LAX_RV_FREEWALL,
    ContourType.SAX_RV_SEPTUM,
    ContourType.LAX_RV_SEPTUM,
    ContourType.SAX_LV_ENDOCARDIAL,
    ContourType.SAX_LV_EPICARDIAL,
    ContourType.RV_INSERT,
    ContourType.APEX_POINT,
    ContourType.MITRAL_VALVE,
    ContourType.TRICUSPID_VALVE,
    ContourType.AORTA_VALVE,
    ContourType.PULMONARY_VALVE,
    ContourType.SAX_RV_EPICARDIAL,
    ContourType.LAX_RV_EPICARDIAL,
    ContourType.LAX_LV_ENDOCARDIAL,
    ContourType.LAX_LV_EPICARDIAL,
    ContourType.LAX_RV_EPICARDIAL,
    ContourType.SAX_RV_OUTLET,
    ContourType.AORTA_PHANTOM,
    ContourType.TRICUSPID_PHANTOM,
    ContourType.MITRAL_PHANTOM,
    ContourType.PULMONARY_PHANTOM,
    ContourType.EXCLUDED,

]

def generate_html(folder: str,  out_dir: str ="./results/", gp_suffix: str ="", si_suffix: str ="", frames_to_fit: list[int]=[], my_logger: logger = logger, model_path = None) -> None:

    # extract the patient name from the folder name
    case = os.path.basename(os.path.normpath(folder))
    my_logger.info(f"case: {case}")

    filename_info = Path(folder) / f"SliceInfoFile{si_suffix}.txt"
    if not filename_info.exists():
        my_logger.error(f"Cannot find {filename_info} file! Skipping this model")
        return -1

    rule = re.compile(fnmatch.translate(f"GPFile_{gp_suffix}*.txt"), re.IGNORECASE)
    time_frame = [Path(folder) / Path(name) for name in os.listdir(Path(folder)) if rule.match(name)]
    frame_name = [re.search(r'GPFile_*(\d+)\.txt', str(file), re.IGNORECASE)[1] for file in time_frame]
    frame_name = sorted(frame_name)

    if len(frames_to_fit) == 0:
        frames_to_fit = np.unique(
            frame_name
        )  # if you want to fit all _frames#

    # create a separate output folder for each patient
    output_folder = Path(out_dir) / case
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    with Progress(transient=True) as progress:
        task = progress.add_task(f"Processing {len(frames_to_fit)} frames", total=len(frames_to_fit))
        console = progress

        for idx, num in enumerate(sorted(frames_to_fit)):
            num = int(num)  # frame number
            filename = Path(folder) / f"GPFile_{gp_suffix}{num:03}.txt"

            if not filename.exists():
                my_logger.error(f"Cannot find {filename} file! Skipping this model")
                return -1

            data_set = GPDataSet(
                str(filename),
                str(filename_info),
                case,
                sampling=1,
                time_frame_number=num,
            )

            if not data_set.success:
                my_logger.error(f"Cannot initialize GPDataSet! Skipping this frame")
                continue

            contour_plots = data_set.plot_dataset(contours_to_plot)
            if model_path is not None:
                rule = re.compile(fnmatch.translate(f"*model_*frame*{num:03}.txt"), re.IGNORECASE)

                path_to_model = [Path(model_path) / name for name in os.listdir(model_path) if rule.match(name)]

                biventricular_model = BiventricularModel(MODEL_RESOURCE_DIR)
                control_points = np.loadtxt(path_to_model[0], delimiter=',', skiprows=1, usecols=[0, 1, 2]).astype(float)
                biventricular_model.update_control_mesh(control_points)

                model = biventricular_model.plot_surface(
                    "rgb(0,127,0)", "rgb(0,127,127)", "rgb(127,0,0)", "all"
                )
                data = contour_plots + model

            else:
                data = contour_plots

            output_folder_html = Path(output_folder, f"html{gp_suffix}")
            output_folder_html.mkdir(exist_ok=True)

            plot(
                go.Figure(data),
                filename=os.path.join(
                    output_folder_html, f"{case}_gp_dataset_frame_{num:03}.html"
                ),
                auto_open=False,
            )

            progress.advance(task)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='This function plots a GPFile  ')
    parser.add_argument('-o', '--output_folder', type=Path, default="./html",
                        help='Path to the output folder')
    parser.add_argument('-gp', '--gp_directory', type=Path, 
                        help='Define the directory containing guidepoint files', default="./html")
    parser.add_argument('--gp_suffix', type =str, default = '', help='guidepoints to use if we do not want to fit all the models in the input folder')
    parser.add_argument('--si_suffix', type =str, default = '', help='Define slice info to use if multiple SliceInfo.txt file are available')
    parser.add_argument('-mdir', '--model_directory', type=Path,
                        help='Define the directory containing the model files', default = None)

    args = parser.parse_args()

    # save config file to the output folder
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if args.model_directory is not None:
        assert Path(args.model_directory).exists(), \
            f'model_directory does not exist. Cannot find {args.model_directory}!'

    # set list of cases to process
    case_list = os.listdir(args.gp_directory)
    case_dirs = [Path(args.gp_directory, case).as_posix() for case in case_list if not case.startswith('.')]
    logger.info(f"Found {len(case_dirs)} cases to plot.")

    # start processing...
    start_time = time.time()

    try:
        for case in case_dirs:
            try:
                logger.info(f"Processing {os.path.basename(case)}")
                if args.model_directory is not None:
                    model_dir = Path(args.model_directory) / os.path.basename(case)
                else:
                    model_dir = None
                generate_html(case, out_dir=output_folder, gp_suffix=args.gp_suffix, si_suffix=args.si_suffix,
                                frames_to_fit=[], my_logger=logger, model_path = model_dir)
            except:
                logger.error(f"Could not process: {os.path.basename(case)}")

        logger.info(f"Total cases processed: {len(case_dirs)}")
        logger.info(f"Total time: {time.time() - start_time}")
        logger.success(f'Done. Results are saved in {output_folder}')

    except KeyboardInterrupt:
        logger.info(f"Program interrupted by the user")
        sys.exit(0)