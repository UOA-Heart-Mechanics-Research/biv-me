import os, sys
import numpy as np
import time
import pandas as pd
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
from copy import deepcopy
from bivme.fitting.surface_enum import Surface, ControlMesh

from bivme.fitting.BiventricularModel import BiventricularModel
from bivme.fitting.GPDataSet import GPDataSet
from bivme.fitting.surface_enum import ContourType
from bivme.fitting.diffeomorphic_fitting_utils import (
    solve_least_squares_problem,
    solve_convex_problem,
    plot_timeseries,
)

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
    ContourType.EXCLUDED,
]

def perform_fitting(folder: str,  config: dict, out_dir: str ="./results/", gp_suffix: str ="", si_suffix: str ="", frames_to_fit: list[int]=[], output_format: str =".vtk", my_logger: logger = logger, **kwargs) -> float:
    # performs all the BiVentricular fitting operations

    try:
        #if "iter_num" in kwargs:
        #    iter_num = kwargs.get("iter_num", None)
        #    pid = os.getpid()
        #    # print('child PID', pid)
        #    # assign a new process ID and a new CPU to the child process
        #    # iter_num corresponds to the id number of the CPU where the process will be run
        #    os.system("taskset -cp %d %d" % (iter_num, pid))

        if "id_Frame" in kwargs:
            # acquire .csv file containing patient_id, ES frame number, ED frame number if present
            case_frame_dict = kwargs.get("id_Frame", None)

        filename_info = Path(folder) / f"SliceInfoFile{si_suffix}.txt"
        if not filename_info.exists():
            my_logger.error(f"Cannot find {filename_info} file! Skipping this model")
            return -1

        # extract the patient name from the folder name
        case = os.path.basename(os.path.normpath(folder))
        my_logger.info(f"case: {case}")

        rule = re.compile(fnmatch.translate(f"GPFile_{gp_suffix}*.txt"), re.IGNORECASE)
        time_frame = [Path(folder) / Path(name) for name in os.listdir(Path(folder)) if rule.match(name)]
        frame_name = [re.search(r'GPFile_*(\d+)\.txt', str(file), re.IGNORECASE)[1] for file in time_frame]
        frame_name = sorted(frame_name)

        ed_frame = config["breathhold_correction"]["ed_frame"]
        my_logger.info(f'ED set to frame #{config["breathhold_correction"]["ed_frame"]}')

        if len(frames_to_fit) == 0:
            frames_to_fit = np.unique(
                frame_name
            )  # if you want to fit all _frames#

        # create a separate output folder for each patient
        output_folder = Path(out_dir) / case
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # create log files where to store fitting errors and shift
        shift_file = output_folder / f"shift_file{gp_suffix}.txt"
        pos_file = output_folder / f"pos_file{gp_suffix}.txt"

        # The next lines are used to measure shift using only a key frame
        if config["breathhold_correction"]["shifting"] == "derived_from_ed":
            my_logger.info("Shift measured only at ED frame")
            filename = Path(folder) / f"GPFile_{gp_suffix}{frame_name[ed_frame]:03}.txt"

            if not filename.exists():
                my_logger.error(f"Cannot find {filename} file! Skipping this model")
                return -1

            ed_dataset = GPDataSet(
                str(filename),
                str(filename_info),
                case,
                sampling=config["gp_processing"]["sampling"],
                time_frame_number=ed_frame,
            )
            if not ed_dataset.success:
                return -1
            result_at_ed = ed_dataset.sinclair_slice_shifting(my_logger)
            _, _ = ed_dataset.get_unintersected_slices()

            ##TODO remove basal slice (maybe looking at the distance between the contours centroid and the projection of the line mitral centroid/apex)
            shift_to_apply = result_at_ed[0]
            updated_slice_position = result_at_ed[1]

            with shift_file.open("w", encoding ="utf-8") as file:
                file.write("shift measured only at ED: frame " + str(ed_frame) + "\n")
                file.write(str(shift_to_apply))
                file.close()

            with pos_file.open("w", encoding ="utf-8") as file:
                file.write("pos measured only at ED: frame " + str(ed_frame) + "\n")
                file.write(str(updated_slice_position))
                file.close()

        elif config["breathhold_correction"]["shifting"] == "average_all_frames":
            my_logger.info("Shift measured on all the frames and averaged")
            shift_to_apply = 0  # 2D translation
            updated_slice_position = 0
            counter = 0
            for frame in sorted(frames_to_fit):
                num = int(frame)
                filename = Path(folder) / f"GPFile_{gp_suffix}{num:03}.txt"
                if not filename.exists():
                    my_logger.error(f"Cannot find {filename} file! Skipping this model")
                    return -1

                dataset = GPDataSet(
                    str(filename),
                    str(filename_info),
                    case,
                    sampling=config["gp_processing"]["sampling"],
                    time_frame_number=num,
                )

                if frame == frames_to_fit[ed_frame]:
                    ed_dataset = deepcopy(dataset)
                if not dataset.success:
                    continue
                result_at_t = dataset.sinclair_slice_shifting(my_logger)

                shift_to_apply += result_at_t[0]
                updated_slice_position += result_at_t[1]
                counter += 1

            shift_to_apply /= counter
            updated_slice_position /= counter

            with shift_file.open("w", encoding ="utf-8") as file:
                file.write("Average shift \n")
                file.write(str(shift_to_apply))
                file.close()

            with pos_file.open("w", encoding ="utf-8") as file:
                file.write("Average shift \n")
                file.write(str(updated_slice_position))
                file.close()

        elif config["breathhold_correction"]["shifting"] == "none":
            my_logger.info("No correction applied")
            filename = Path(folder) / f"GPFile_{gp_suffix}{frame_name[ed_frame]:03}.txt"
            if not filename.exists():
                my_logger.error(f"Cannot find {filename} file! Skipping this model")
                return -1

            ed_dataset = GPDataSet(
                str(filename),
                str(filename_info),
                case,
                sampling=config["gp_processing"]["sampling"],
                time_frame_number=ed_frame,
            )
            if not ed_dataset.success:
                return -1

        else:
            my_logger.error(f'Method {config["breathhold_correction"]["shifting"]} unavailable.  Allowed values: none, derived_from_ed or average_all_frame. No correction applied')
            return -1

        biventricular_model = BiventricularModel(MODEL_RESOURCE_DIR, case)

        my_logger.info(f"Calculating pose and scale {str(case)}...")
        biventricular_model.update_pose_and_scale(ed_dataset)
        aligned_biventricular_model = deepcopy(biventricular_model)

        # initialise time series lists
        my_logger.info(f"Fitting of {str(case)}")

        residuals = 0
        with Progress(transient=True) as progress:
            task = progress.add_task(f"Processing {len(frames_to_fit)} frames", total=len(frames_to_fit))
            console = progress

            for idx, num in enumerate(sorted(frames_to_fit)):
                num = int(num)  # frame number

                my_logger.info(f"Processing frame #{num}")
                model_file = Path(
                    output_folder, f"{case}{gp_suffix}_model_frame_{num:03}.txt"
                )
                model_file.touch(exist_ok=True)

                filename = Path(folder) / f"GPFile_{gp_suffix}{num:03}.txt"
                if not filename.exists():
                    my_logger.error(f"Cannot find {filename} file! Skipping this model")
                    continue

                data_set = GPDataSet(
                    str(filename), str(filename_info), case, sampling=config["gp_processing"]["sampling"], time_frame_number=num
                )
                if not data_set.success:
                    my_logger.error(f"Cannot initialize GPDataSet! Skipping this frame")
                    continue

                if config["breathhold_correction"]["shifting"] != "none":
                    data_set.apply_slice_shift(shift_to_apply, updated_slice_position)
                    data_set.get_unintersected_slices()

                # Generates RV epicardial point if they have not been contoured
                if sum(data_set.contour_type == (ContourType.SAX_RV_EPICARDIAL)) == 0 and sum(data_set.contour_type == ContourType.LAX_RV_EPICARDIAL) == 0:
                    my_logger.info('Generating RV epicardial points')
                    _, _, _ = data_set.create_rv_epicardium(
                        rv_thickness=3
                    )

                try:
                    _ = data_set.create_valve_phantom_points(config["gp_processing"]["num_of_phantom_points_mv"], ContourType.MITRAL_VALVE)
                except:
                    my_logger.warning('Error in creating mitral phantom points')

                try:
                    _ = data_set.create_valve_phantom_points(config["gp_processing"]["num_of_phantom_points_tv"], ContourType.TRICUSPID_VALVE)
                except:
                    my_logger.warning('Error in creating tricuspid phantom points')

                try:
                    _ = data_set.create_valve_phantom_points(config["gp_processing"]["num_of_phantom_points_pv"], ContourType.PULMONARY_VALVE)
                except:
                    my_logger.warning('Error in creating pulmonary phantom points')

                try:
                    _ = data_set.create_valve_phantom_points(config["gp_processing"]["num_of_phantom_points_av"], ContourType.AORTA_VALVE)
                except:
                    my_logger.warning('Error in creating aorta phantom points')

                contour_plots = data_set.plot_dataset(contours_to_plot)

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

                # Perform linear fit
                biventricular_model = deepcopy(aligned_biventricular_model)
                solve_least_squares_problem(biventricular_model, config["fitting_weights"]["guide_points"], data_set, my_logger)
#
                ## Perform diffeomorphic fit
                residuals += solve_convex_problem(
                    biventricular_model,
                    data_set,
                    config["fitting_weights"]["guide_points"],
                    config["fitting_weights"]["convex_problem"],
                    config["fitting_weights"]["transmural"],
                    my_logger,
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

                if output_format != "none":
                    meshes = {}
                    for surface in Surface:
                        mesh_data = {}
                        if surface.name in config["output"]["output_meshes"]:
                            mesh_data[surface.name] = surface.value
                            if surface.name == "LV_ENDOCARDIAL" and config["output"]["closed_mesh"] == True:
                                mesh_data["MITRAL_VALVE"] = Surface.MITRAL_VALVE.value
                                mesh_data["AORTA_VALVE"] = Surface.AORTA_VALVE.value
                            if surface.name == "EPICARDIAL" and config["output"]["closed_mesh"] == True:
                                mesh_data["PULMONARY_VALVE"] = Surface.PULMONARY_VALVE.value
                                mesh_data["TRICUSPID_VALVE"] = Surface.TRICUSPID_VALVE.value
                                mesh_data["MITRAL_VALVE"] = Surface.MITRAL_VALVE.value
                                mesh_data["AORTA_VALVE"] = Surface.AORTA_VALVE.value
                            meshes[surface.name] = mesh_data

                    if "RV_ENDOCARDIAL" in config["output"]["output_meshes"]:
                        mesh_data["RV_SEPTUM"] = Surface.RV_SEPTUM.value
                        mesh_data["RV_FREEWALL"] = Surface.RV_FREEWALL.value
                        if config["output"]["closed_mesh"]:
                            mesh_data["PULMONARY_VALVE"] = Surface.PULMONARY_VALVE.value
                            mesh_data["TRICUSPID_VALVE"] = Surface.TRICUSPID_VALVE.value
                        meshes["RV_ENDOCARDIAL"] = mesh_data

                    ##TODO remove duplicated code here - not sure how yet
                    if config["output"]["export_control_mesh"]:
                        control_mesh_meshes = {}
                        for surface in ControlMesh:
                            control_mesh_mesh_data = {}
                            if surface.name in config["output"]["output_meshes"]:
                                control_mesh_mesh_data[surface.name] = surface.value
                                if surface.name == "LV_ENDOCARDIAL" and config["output"]["closed_mesh"] == True:
                                    control_mesh_mesh_data["MITRAL_VALVE"] = ControlMesh.MITRAL_VALVE.value
                                    control_mesh_mesh_data["AORTA_VALVE"] = ControlMesh.AORTA_VALVE.value
                                if surface.name == "EPICARDIAL" and config["output"]["closed_mesh"] == True:
                                    control_mesh_mesh_data["PULMONARY_VALVE"] = ControlMesh.PULMONARY_VALVE.value
                                    control_mesh_mesh_data["TRICUSPID_VALVE"] = ControlMesh.TRICUSPID_VALVE.value
                                    control_mesh_mesh_data["MITRAL_VALVE"] = ControlMesh.MITRAL_VALVE.value
                                    control_mesh_mesh_data["AORTA_VALVE"] = ControlMesh.AORTA_VALVE.value
                                if surface.name == "RV_ENDOCARDIAL" and config["output"]["closed_mesh"] == True:
                                    control_mesh_mesh_data["PULMONARY_VALVE"] = ControlMesh.PULMONARY_VALVE.value
                                    control_mesh_mesh_data["TRICUSPID_VALVE"] = ControlMesh.TRICUSPID_VALVE.value

                                control_mesh_meshes[surface.name] = control_mesh_mesh_data

                    for key, value in meshes.items():
                        vertices = np.array([]).reshape(0, 3)
                        faces_mapped = np.array([], dtype=np.int64).reshape(0, 3)

                        offset = 0
                        for type in value:
                            start_fi = biventricular_model.surface_start_end[value[type]][0]
                            end_fi = biventricular_model.surface_start_end[value[type]][1] + 1
                            faces_et = biventricular_model.et_indices[start_fi:end_fi]
                            unique_inds = np.unique(faces_et.flatten())
                            vertices = np.vstack((vertices, biventricular_model.et_pos[unique_inds]))

                            # remap faces/indices to 0-indexing
                            mapping = {old_index: new_index for new_index, old_index in enumerate(unique_inds)}
                            faces_mapped = np.vstack((faces_mapped, np.vectorize(mapping.get)(faces_et) + offset))
                            offset += len(biventricular_model.et_pos[unique_inds])

                        if output_format == ".vtk":
                            output_folder_vtk = Path(output_folder, f"vtk{gp_suffix}")
                            output_folder_vtk.mkdir(exist_ok=True)
                            mesh_path = Path(
                                output_folder_vtk, f"{case}_{key}_{num:03}.vtk"
                            )
                            write_vtk_surface(str(mesh_path), vertices, faces_mapped)
                            my_logger.success(f"{case}_{key}_{num:03}.vtk successfully saved to {output_folder_vtk}")

                        elif output_format == ".obj":
                            output_folder_obj = Path(output_folder, f"obj{gp_suffix}")
                            output_folder_obj.mkdir(exist_ok=True)
                            mesh_path = Path(
                                output_folder_obj, f"{case}_{key}_{num:03}.obj"
                            )
                            export_to_obj(mesh_path, vertices, faces_mapped)
                            my_logger.success(f"{case}_{key}_{num:03}.obj successfully saved to {output_folder_obj}")
                        else:
                            my_logger.error('argument format must be .obj or .vtk')
                            return -1

                    ##TODO remove duplicated code here - not sure how yet
                    if config["output"]["export_control_mesh"]:
                        for key, value in control_mesh_meshes.items():
                            vertices = np.array([]).reshape(0, 3)
                            faces_mapped = np.array([], dtype=np.int64).reshape(0, 3)

                            offset = 0
                            for type in value:
                                start_fi = biventricular_model.control_mesh_start_end[value[type]][0]
                                end_fi = biventricular_model.control_mesh_start_end[value[type]][1] + 1
                                faces_et = biventricular_model.et_indices_control_mesh[start_fi:end_fi]
                                unique_inds = np.unique(faces_et.flatten())
                                vertices = np.vstack((vertices, biventricular_model.control_mesh[unique_inds]))

                                # remap faces/indices to 0-indexing
                                mapping = {old_index: new_index for new_index, old_index in enumerate(unique_inds)}
                                faces_mapped = np.vstack((faces_mapped, np.vectorize(mapping.get)(faces_et) + offset))
                                offset += len(biventricular_model.control_mesh[unique_inds])

                            if output_format == ".vtk":
                                output_folder_vtk = Path(output_folder, f"vtk{gp_suffix}")
                                output_folder_vtk.mkdir(exist_ok=True)
                                mesh_path = Path(
                                    output_folder_vtk, f"{case}_{key}_{num:03}_control_mesh.vtk"
                                )
                                write_vtk_surface(str(mesh_path), vertices, faces_mapped)
                                my_logger.success(f"{case}_{key}_{num:03}_control_mesh.vtk successfully saved to {output_folder_vtk}")

                            elif output_format == ".obj":
                                output_folder_obj = Path(output_folder, f"obj{gp_suffix}")
                                output_folder_obj.mkdir(exist_ok=True)
                                mesh_path = Path(
                                    output_folder_obj, f"{case}_{key}_{num:03}_control_mesh.obj"
                                )
                                export_to_obj(mesh_path, vertices, faces_mapped)
                                my_logger.success(f"{case}_{key}_{num:03}_control_mesh.obj successfully saved to {output_folder_obj}")
                            else:
                                my_logger.error('argument format must be .obj or .vtk')
                                return -1

                progress.advance(task)
        return residuals
    except KeyboardInterrupt:
        return -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Biv-me')
    parser.add_argument('-config', '--config_file', type=str,
                        help='Config file containing fitting parameters')
    args = parser.parse_args()

    # Load config
    assert Path(args.config_file).exists(), \
        f'Cannot not find {args.config_file}!'
    with open(args.config_file, mode="rb") as fp:
        config = tomli.load(fp)

    # TOML Schema Validation
    match config:
        case {
            "input": {"gp_directory": str(),
                      "gp_suffix": str(),
                      "si_suffix": str(),
                      },
            "breathhold_correction": {"shifting": str(), "ed_frame": int()},
            "gp_processing": {"sampling": int(), "num_of_phantom_points_av": int(), "num_of_phantom_points_mv": int(), "num_of_phantom_points_tv": int(), "num_of_phantom_points_pv": int()},
            "multiprocessing": {"workers": int()},
            "fitting_weights": {"guide_points": float(), "convex_problem": float(), "transmural": float()},
            "output": {"output_directory": str(), "output_meshes": list(), "closed_mesh": bool(),  "show_logging": bool(), "export_control_mesh": bool(), "mesh_format": str(), "generate_log_file": bool(), "overwrite": bool()},
        }:
            pass
        case _:
            raise ValueError(f"Invalid configuration: {config}")

    # save config file to the output folder
    output_folder = Path(config["output"]["output_directory"])
    output_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config_file, output_folder)

    assert Path(config["input"]["gp_directory"]).exists(), \
        f'gp_directory does not exist. Cannot find {config["input"]["gp_directory"]}!'

    # set list of cases to process
    case_list = os.listdir(config["input"]["gp_directory"])
    case_dirs = [Path(config["input"]["gp_directory"], case).as_posix() for case in case_list]

    if not config["output"]["show_logging"]:
        logger.remove()

    log_level = "DEBUG"
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"

    logger.info(f"Found {len(case_dirs)} cases to fit.")
    # start processing...
    start_time = time.time()

    if not (config["output"]["mesh_format"].endswith('.obj') or config["output"]["mesh_format"].endswith('.vtk') or config["output"]["mesh_format"] == 'none'):
        logger.error(f'argument mesh_format must be .obj, .vtk or none. {config["output"]["mesh_format"]} given.')
        sys.exit(0)

    for mesh in config["output"]["output_meshes"]:
        if mesh not in ["LV_ENDOCARDIAL", "RV_ENDOCARDIAL", "EPICARDIAL"]:
            logger.error(f'argument output_meshes invalid. {mesh} given. Allowed values are "LV_ENDOCARDIAL", "RV_ENDOCARDIAL", "EPICARDIAL"')
            sys.exit(0)

    try:
        for case in case_dirs:
            logger.info(f"Processing {os.path.basename(case)}")
            if config["output"]["generate_log_file"]:
                logger_id = logger.add(f'{config["output"]["output_directory"]}/{os.path.basename(case)}/log_file_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log', level=log_level, format=log_format,
                                            colorize=False, backtrace=True,
                                            diagnose=True)

            if not config["output"]["overwrite"]:
                rule = re.compile(fnmatch.translate(f"{os.path.basename(case)}_model_frame*.txt"), re.IGNORECASE)
                case_folder = os.path.join(config["output"]["output_directory"], os.path.basename(case))
                cases = [name for name in os.listdir(Path(case_folder)) if rule.match(name)]
                if len(cases) > 0:
                    logger.info("Folder already exists for this case. Proceeding to next case")
                    continue

            residuals = perform_fitting(case, config, out_dir=config["output"]["output_directory"], gp_suffix=config["input"]["gp_suffix"], si_suffix=config["input"]["si_suffix"],
                            frames_to_fit=[], output_format=config["output"]["mesh_format"], logger=logger)
            logger.info(f"Average residuals: {residuals} for case {os.path.basename(case)}")
            if config["output"]["generate_log_file"]:
                logger.remove(logger_id)

        logger.info(f"Total cases processed: {len(case_dirs)}")
        logger.info(f"Total time: {time.time() - start_time}")
        logger.success(f'Done. Results are saved in {config["output"]["output_directory"]}')
    except KeyboardInterrupt:
        logger.info(f"Program interrupted by the user")
        sys.exit(0)