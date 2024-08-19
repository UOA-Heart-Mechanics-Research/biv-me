import os, sys
import numpy as np
import time
import pandas as pd
import plotly.graph_objs as go
from pathlib import Path
from plotly.offline import plot
import argparse
import pathlib

from bivme.fitting.BiventricularModel import BiventricularModel
from bivme.fitting.GPDataSet import GPDataSet
from bivme.fitting.surface_enum import ContourType
from bivme.fitting.diffeomorphic_fitting_utils import (
    solve_least_squares_problem,
    solve_convex_problem,
    plot_timeseries,
)

from bivme.fitting.config_params import *
from bivme.meshing.mesh_io import write_vtk_surface, export_to_obj
from loguru import logger
from rich.progress import Progress
from bivme import MODEL_RESOURCE_DIR

# This list of contours_to _plot was taken from Liandong Lee
contours_to_plot = [
    ContourType.LAX_RA,
    ContourType.LAX_LA,
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
]

def perform_fitting(folder: str, out_dir: str ="./results/", gp_suffix: str ="", si_suffix: str ="", frames_to_fit: list[int]=[], output_format: str =".vtk", **kwargs) -> float:
    # performs all the BiVentricular fitting operations

    try:
        if "iter_num" in kwargs:
            iter_num = kwargs.get("iter_num", None)
            pid = os.getpid()
            # print('child PID', pid)
            # assign a new process ID and a new CPU to the child process
            # iter_num corresponds to the id number of the CPU where the process will be run
            os.system("taskset -cp %d %d" % (iter_num, pid))

        if "id_Frame" in kwargs:
            # acquire .csv file containing patient_id, ES frame number, ED frame number if present
            case_frame_dict = kwargs.get("id_Frame", None)

        filename = Path(folder) / f"GPFile{gp_suffix}.txt"
        if not filename.exists():
            logger.error(f"Cannot find {filename} file! Skipping this model")
            return -1

        filename_info = Path(folder) / f"SliceInfoFile{si_suffix}.txt"
        if not filename_info.exists():
            logger.error(f"Cannot find {filename_info} file! Skipping this model")
            return -1

        # extract the patient name from the folder name
        case = os.path.basename(os.path.normpath(folder))
        logger.info(f"case: {case}")

        # read all the frames from the GPFile
        all_frames = pd.read_csv(str(filename), sep="\t")
        # select which frames to fit
        try:
            ed_frame = int(case_frame_dict[str(case)][0])
        except:
            ed_frame = 1
            logger.info(f"ED set to frame # 1")

        if len(frames_to_fit) == 0:
            frames_to_fit = np.unique(
                [i[6] for i in all_frames.values]
            )  # if you want to fit all _frames

        # create a separate output folder for each patient
        output_folder = Path(out_dir) / case
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # create log files where to store fitting errors and shift
        error_file = output_folder / f"error_file{gp_suffix}.txt"
        shift_file = output_folder / f"shift_file{gp_suffix}.txt"
        pos_file = output_folder / f"pos_file{gp_suffix}.txt"

        with error_file.open("w", encoding ="utf-8") as f: #automatically initialises it
            f.write("Log for patient: " + case + "\n")

        # The next lines are used to measure shift using only a key frame
        if measure_shift_EDonly == True:
            logger.info("Shift measured only at ED frame")

            ed_dataset = GPDataSet(
                str(filename),
                str(filename_info),
                case,
                sampling=sampling,
                time_frame_number=ed_frame,
            )
            result_at_ed = ed_dataset.sinclaire_slice_shifting(frame_num=ed_frame)
            shift_at_ed = result_at_ed[0]
            pos_ed = result_at_ed[1]
            # np.save(os.path.join(output_folder, 'shift.txt'), shift_at_ed)
            with shift_file.open("w", encoding ="utf-8") as file:
                file.write("shift measured only at ED: frame " + str(ed_frame) + "\n")
                file.write(str(shift_at_ed))
                file.close()

            with pos_file.open("w", encoding ="utf-8") as file:
                file.write("pos measured only at ED: frame " + str(ed_frame) + "\n")
                file.write(str(pos_ed))
                file.close()

        # initialise time series lists
        logger.info(f"Fitting of {str(case)}")

        residuals = 0
        with Progress(transient=True) as progress:
            task = progress.add_task(f"Processing {len(frames_to_fit)} frames", total=len(frames_to_fit))
            console = progress

            for idx, num in enumerate(sorted(frames_to_fit)):
                num = int(num)  # frame number
                if logger is not None:
                    logger.info(f"Processing frame #{num}")
                model_file = Path(
                    output_folder, f"{case}{gp_suffix}_model_frame_{num:03}.txt"
                )
                model_file.touch(exist_ok=True)

                with open(error_file, "a") as f:
                    f.write("\nFRAME #" + str(int(num)) + "\n")

                data_set = GPDataSet(
                    str(filename), str(filename_info), case, sampling=sampling, time_frame_number=num
                )
                biventricular_model = BiventricularModel(MODEL_RESOURCE_DIR, case)
                model = biventricular_model.plot_surface(
                    "rgb(0,127,0)", "rgb(0,0,127)", "rgb(127,0,0)", surface="all"
                )

                if measure_shift_EDonly == True:
                    # apply shift measured previously using ED frame
                    data_set.apply_slice_shift(shift_at_ed, pos_ED)
                    data_set.get_unintersected_slices()
                else:
                    # measure and apply shift to current frame
                    shifted_slice = data_set.sinclaire_slice_shifting(error_file, int(num))
                    shift_measure = shifted_slice[0]
                    pos_measure = shifted_slice[1]

                    if idx == 0:
                        with open(shift_file, "w") as file:
                            file.write("Frame number:  " + str(num) + "\n")
                            file.write(str(shift_measure))
                            file.close()

                        with open(pos_file, "w") as file:
                            file.write("Frame number:  " + str(num) + "\n")
                            file.write(str(pos_measure))
                            file.close()

                    else:
                        with open(shift_file, "a") as file:
                            file.write("\nFrame number:  " + str(num) + "\n")
                            file.write(str(shift_measure))
                            file.close()

                        with open(pos_file, "w") as file:
                            file.write("\nFrame number:  " + str(num) + "\n")
                            file.write(str(pos_measure))
                            file.close()
                    pass

                biventricular_model.update_pose_and_scale(data_set)

                # # perform a stiff fit
                # displacement, err = biventricular_model.lls_fit_model(weight_GP,data_set,1e10)
                # biventricular_model.control_mesh = np.add(biventricular_model.control_mesh,
                #                                           displacement)
                # biventricular_model.et_pos = np.linalg.multi_dot([biventricular_model.matrix,
                #                                                   biventricular_model.control_mesh])
                # displacements = data_set.SAXSliceShiffting(biventricular_model)

                # contour_plots = data_set.PlotDataSet(contours_to_plot)

                # plot(go.Figure(contour_plots))
                # data = contour_plots

                # plot(go.Figure(data),filename=os.path.join(folder, 'pose_fitted_model_Frame'+str(int(num))+'.html'), auto_open=False)

                # Generates RV epicardial point if they have not been contoured
                # (can be commented if available) used in LL

                _, _, _ = data_set.create_rv_epicardium(
                    rv_thickness=3
                )

                # Generates 30 BP_point phantom points and 30 tricuspid phantom points.
                # We do not have any pulmonary points or aortic points in our dataset but if you do,
                # I recommend you to do the same.

                try:
                    _ = data_set.create_valve_phantom_points(20, ContourType.MITRAL_VALVE)
                except:
                    logger.warning('Error in creating mitral phantom points')

                try:
                    _ = data_set.create_valve_phantom_points(20, ContourType.TRICUSPID_VALVE)
                except:
                    logger.warning('Error in creating tricuspid phantom points')

                try:
                    _ = data_set.create_valve_phantom_points(20, ContourType.PULMONARY_VALVE)
                except:
                    logger.warning('Error in creating pulmonary phantom points')

                try:
                    _ = data_set.create_valve_phantom_points(20, ContourType.AORTA_VALVE)
                except:
                    logger.warning('Error in creating aorta phantom points')

                contour_plots = data_set.PlotDataSet(contours_to_plot)

                # Example on how to set different weights for different points group (R.B.)
                data_set.weights[data_set.contour_type == ContourType.MITRAL_PHANTOM] = 2
                data_set.weights[data_set.contour_type == ContourType.AORTA_PHANTOM] = 1
                data_set.weights[data_set.contour_type == ContourType.PULMONARY_PHANTOM] = 1
                data_set.weights[data_set.contour_type == ContourType.TRICUSPID_PHANTOM] = 1

                data_set.weights[data_set.contour_type == ContourType.APEX_POINT] = 1
                data_set.weights[data_set.contour_type == ContourType.RV_INSERT] = 1

                data_set.weights[data_set.contour_type == ContourType.MITRAL_VALVE] = 1
                data_set.weights[data_set.contour_type == ContourType.AORTA_VALVE] = 1
                data_set.weights[data_set.contour_type == ContourType.PULMONARY_VALVE] = 1

                # Perform linear fit (step1)
                solve_least_squares_problem(biventricular_model, weight_GP, data_set, error_file)

                # Perform diffeomorphic fit (step2)
                residuals += solve_convex_problem(
                    biventricular_model,
                    data_set,
                    weight_GP,
                    low_smoothing_weight,
                    transmural_weight,
                    error_file,
                ) / len(sorted(frames_to_fit))

                # Plot final results
                model = biventricular_model.plot_surface(
                    "rgb(0,127,0)", "rgb(0,127,127)", "rgb(127,0,0)", "all"
                )

                data = contour_plots + model

                output_folder_html = Path(output_folder, f"html{gp_suffix}")
                output_folder_html.mkdir(exist_ok=True)
                plot(
                    go.Figure(data),
                    filename=os.path.join(
                        output_folder_html, f"{case}_fitted_model_frame_{num:03}.html"
                    ),
                    auto_open=False,
                )

                # save results in .txt format, one file for each frame
                model_data = {
                    "x": biventricular_model.control_mesh[:, 0],
                    "y": biventricular_model.control_mesh[:, 1],
                    "z": biventricular_model.control_mesh[:, 2],
                    "Frame": [num] * len(biventricular_model.control_mesh[:, 2]),
                }

                model_data_frame = pd.DataFrame(data=model_data)
                with open(model_file, "w") as file:
                    file.write(
                        model_data_frame.to_csv(
                            header=True, index=False, sep=",", lineterminator="\n"
                        )
                    )

                if output_format == ".obj":
                    output_folder_obj = Path(output_folder, f"obj{gp_suffix}")
                    output_folder_obj.mkdir(exist_ok=True)
                    vertices = biventricular_model.et_pos
                    faces = biventricular_model.et_indices
                    output_path = Path(
                        output_folder_obj, f"{case}_{num:03}.obj"
                    )
                    export_to_obj(output_path, vertices, faces)
                    logger.success(f"{case}_{num:03}.obj successfully saved to {output_folder_obj}")
                elif output_format == ".vtk":
                    # save surface meshes as vtk
                    output_folder_vtk = Path(output_folder, f"vtk{gp_suffix}")
                    output_folder_vtk.mkdir(exist_ok=True)
                    mesh_type = ["Biv", "LV_endocardium", "RV_endocardium", "epicardium"]
                    mesh_data = [{"LV_endocardium": 0, "RV_septum": 1, "RV_freewall": 2, "epicardium": 3},
                                 {"LV_endocardium": 0},
                                 {"RV_septum": 1, "RV_freewall": 2},
                                 {"epicardium": 3}]

                    for i, mesh in enumerate(mesh_data):
                        vertices = np.array([]).reshape(0, 3)
                        faces_mapped = np.array([], dtype=np.int64).reshape(0, 3)

                        offset = 0
                        for type in mesh:
                            start_fi = biventricular_model.surface_start_end[mesh[type]][0]
                            end_fi = biventricular_model.surface_start_end[mesh[type]][1] + 1
                            faces_et = biventricular_model.et_indices[start_fi:end_fi]
                            unique_inds = np.unique(faces_et.flatten())
                            vertices = np.vstack((vertices, biventricular_model.et_pos[unique_inds]))

                            # remap faces/indices to 0-indexing
                            mapping = {old_index: new_index for new_index, old_index in enumerate(unique_inds)}
                            faces_mapped = np.vstack((faces_mapped, np.vectorize(mapping.get)(faces_et) + offset))
                            offset += len(biventricular_model.et_pos[unique_inds])

                        mesh_path = Path(
                            output_folder_vtk, f"{case}_{mesh_type[i]}_{num:03}.vtk"
                        )
                        write_vtk_surface(str(mesh_path), vertices, faces_mapped)
                        logger.success(f"{case}_{mesh_type[i]}_{num:03}.vtk successfully saved to {output_folder_vtk}")

                else:
                    logger.error('argument format must be .obj or .vtk')
                    return -1

                progress.advance(task)
        return residuals
    except KeyboardInterrupt:
        return -1


if __name__ == "__main__":
    ##TODO create json config for setup and save this json config so we know the parameters used for each fit
    parser = argparse.ArgumentParser(description='Biv-me')
    parser.add_argument('-gp', '--dir_gp', default="./../../../example/guidepoints", type=str,
                        help='directory containing guidepoint files')
    parser.add_argument('-o', '--output_dir', default="./../../../output/fitted", type=str,
                        help='output directory')
    parser.add_argument('-w', '--overwrite', action="store_true",
                        help='Overwrite existing output mesh')
    parser.add_argument('-gp_suf', '--gp_suffix', default="", type=str,
                        help='guidepoint to use')
    parser.add_argument('-si_suf', '--si_suffix', default="", type=str,
                        help='slice info to use')
    parser.add_argument('-f', '--format', default=".vtk", type=str,
                        help='Format of the output model (Only .obj and .vtk are currently supported)')
    args = parser.parse_args()

    # set list of cases to process
    case_list = os.listdir(args.dir_gp)
    case_dirs = [Path(args.dir_gp, case).as_posix() for case in case_list]

    log_level = "DEBUG"
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"

    logger.info(f"Found {len(case_dirs)} cases to fit.")
    # start processing...
    start_time = time.time()

    if not (args.format.endswith('.obj') or args.format.endswith('.vtk')):
        logger.error('argument format must be .obj or .vtk')
        raise argparse.ArgumentTypeError('argument format must be .obj or .vtk')

    for case in case_dirs:
        logger.info(f"Processing {os.path.basename(case)}")
        logger_id = logger.add(f"{args.output_dir}/{os.path.basename(case)}/log_file.log", level=log_level, format=log_format,
                                    colorize=False, backtrace=True,
                                    diagnose=True)

        if not args.overwrite and os.path.exists(os.path.join(args.output_dir, os.path.basename(case))):
            logger.info("Folder already exists for this case. Proceeding to next case")
            continue

        residuals = perform_fitting(case, out_dir=args.output_dir, gp_suffix=args.gp_suffix, si_suffix=args.si_suffix,
                        frames_to_fit=[], output_format=args.format)
        logger.info(f"Average residuals: {residuals} for case {os.path.basename(case)}")
        logger.remove(logger_id)

    logger.info(f"Total cases processed: {len(case_dirs)}")
    logger.info(f"Total time: {time.time() - start_time}")
    logger.success(f"Done. Results are saved in {args.output_dir}")