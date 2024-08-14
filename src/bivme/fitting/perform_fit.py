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

def perform_fitting(folder: str, out_dir: str ="./results/", gp_suffix: str ="", si_suffix: str ="", frames_to_fit: list=None, output_format: str =".vtk", logger: logger=None, **kwargs) -> None:
    # performs all the BiVentricular fitting operations

    console = None
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
        assert filename.exists(), \
            f"Cannot find {filename} file!"

        filename_info = Path(folder) / f"SliceInfoFile{si_suffix}.txt"
        assert filename_info.exists(), \
            f"Cannot find {filename_info} file!"

        # extract the patient name from the folder name
        case = os.path.basename(os.path.normpath(folder))
        if logger is not None:
            logger.info(f"case: {case}")

        # read all the frames from the GPFile
        all_frames = pd.read_csv(str(filename), sep="\t")
        # select which frames to fit
        try:
            ed_frame = int(case_frame_dict[str(case)][0])
        except:
            ed_frame = 1
            if logger is not None:
                logger.info(f"ED set to frame # 1")

        if frames_to_fit is None:
            frames_to_fit = np.unique(
                [i[6] for i in all_frames.values]
            )  # if you want to fit all _frames

        # path to model template files
        model_path = Path(os.path.dirname(__file__)) / "../../model"

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
            if logger is not None:
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
            pos_ED = result_at_ed[1]
            # np.save(os.path.join(output_folder, 'shift.txt'), shift_at_ed)
            with shift_file.open("w", encoding ="utf-8") as file:
                file.write("shift measured only at ED: frame " + str(ed_frame) + "\n")
                file.write(str(shift_at_ed))
                file.close()

            with pos_file.open("w", encoding ="utf-8") as file:
                file.write("pos measured only at ED: frame " + str(ed_frame) + "\n")
                file.write(str(pos_ED))
                file.close()

        # initialise time series lists
        TimeSeries_step1 = []
        TimeSeries_step2 = []
        if logger is not None:
            logger.info(f"Fitting of {str(case)}")

        with Progress(transient=True) as progress:
            task = progress.add_task(f"Processing {len(frames_to_fit)} frames", total=len(frames_to_fit))
            console = progress

            for idx, num in enumerate(sorted(frames_to_fit)):
                num = int(num)  # frame number
                if logger is not None:
                    logger.info(f"Processing frame #{num}")
                Modelfile = Path(
                    output_folder, f"{case}{gp_suffix}_model_frame_{num:03}.txt"
                )
                Modelfile.touch(exist_ok=True)

                with open(error_file, "a") as f:
                    f.write("\nFRAME #" + str(int(num)) + "\n")

                data_set = GPDataSet(
                    str(filename), str(filename_info), case, sampling=sampling, time_frame_number=num
                )
                biventricular_model = BiventricularModel(model_path, case)
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

                # contourPlots = data_set.PlotDataSet(contours_to_plot)

                # plot(go.Figure(contourPlots))
                # data = contourPlots

                # plot(go.Figure(data),filename=os.path.join(folder, 'pose_fitted_model_Frame'+str(int(num))+'.html'), auto_open=False)

                # Generates RV epicardial point if they have not been contoured
                # (can be commented if available) used in LL

                rv_epi_points, rv_epi_contour, rv_epi_slice = data_set.create_rv_epicardium(
                    rv_thickness=3
                )

                # Generates 30 BP_point phantom points and 30 tricuspid phantom points.
                # We do not have any pulmonary points or aortic points in our dataset but if you do,
                # I recommend you to do the same.

                try:
                    mitral_points = data_set.create_valve_phantom_points(20, ContourType.MITRAL_VALVE)
                except:
                    print('Error in creating mitral phantom points')

                try:
                    tri_points = data_set.create_valve_phantom_points(20, ContourType.TRICUSPID_VALVE)

                except:
                    print('Error in creating tricuspid phantom points')

                try:
                    pulmonary_points = data_set.create_valve_phantom_points(20, ContourType.PULMONARY_VALVE)

                except:
                    print('Error in creating pulmonary phantom points')

                try:
                    aorta_points = data_set.create_valve_phantom_points(20, ContourType.AORTA_VALVE)

                except:
                    print('Error in creating aorta phantom points')
                    pass

                contourPlots = data_set.PlotDataSet(contours_to_plot)

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

                # Plot results
                # model = biventricular_model.plot_surface(
                #     "rgb(0,127,0)", "rgb(0,127,127)", "rgb(127,0,0)", "all"
                # )
                # data = model + contourPlots
                # TimeSeries_step1.append([data, num])

                # Perform diffeomorphic fit (step2)
                solve_convex_problem(
                    biventricular_model,
                    data_set,
                    weight_GP,
                    low_smoothing_weight,
                    transmural_weight,
                    error_file,
                )

                # Plot final results
                model = biventricular_model.plot_surface(
                    "rgb(0,127,0)", "rgb(0,127,127)", "rgb(127,0,0)", "all"
                )
                data = model + contourPlots
                # TimeSeries_step2.append([data, num])

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
                ModelData = {
                    "x": biventricular_model.control_mesh[:, 0],
                    "y": biventricular_model.control_mesh[:, 1],
                    "z": biventricular_model.control_mesh[:, 2],
                    "Frame": [num] * len(biventricular_model.control_mesh[:, 2]),
                }

                Model_Dataframe = pd.DataFrame(data=ModelData)
                with open(Modelfile, "w") as file:
                    file.write(
                        Model_Dataframe.to_csv(
                            header=True, index=False, sep=",", lineterminator="\n"
                        )
                    )

                output_folder_obj = Path(output_folder, f"obj{gp_suffix}")
                output_folder_obj.mkdir(exist_ok=True)

                if output_format == ".obj":
                    vertices = biventricular_model.et_pos
                    faces = biventricular_model.et_indices
                    output_path = Path(
                        output_folder_obj, f"{case}_{num:03}.obj"
                    )
                    if logger is not None:
                        logger.info(f"Saving model to {str(output_path)}")
                    export_to_obj(output_path, vertices, faces)
                    if logger is not None:
                        logger.success(f"Model successfully saved to {output_path}")

                if output_format == ".vtk":
                    # save surface meshes as vtk
                    output_folder_vtk = Path(output_folder, f"vtk{gp_suffix}")
                    output_folder_vtk.mkdir(exist_ok=True)
                    mesh_type = ["epicardium", "LV_endocardium", "RV_freewall", "RV_septum"]
                    mesh_data = {"LV_endocardium": 0, "RV_septum": 1, "RV_freewall": 2, "epicardium": 3}
                    for i in range(4):
                        vertices = np.vstack((data[i].x, data[i].y, data[i].z)).transpose()
                        faces = np.vstack((data[i].i, data[i].j, data[i].k)).transpose()
                        meshpath = Path(
                            output_folder_vtk, f"{case}_{mesh_type[i]}_{num:03}.vtk"
                        )
                        write_vtk_surface(meshpath, vertices, faces)

                        start_fi = biventricular_model.surface_start_end[mesh_data[mesh_type[i]]][0]
                        end_fi = biventricular_model.surface_start_end[mesh_data[mesh_type[i]]][1] + 1
                        faces_et = biventricular_model.et_indices[start_fi:end_fi]

                        unique_inds = np.unique(faces_et.flatten())
                        vertices = biventricular_model.et_pos[unique_inds]

                        # remap faces/indices to 0-indexing
                        mapping = {old_index: new_index for new_index, old_index in enumerate(unique_inds)}
                        faces_mapped = np.vectorize(mapping.get)(faces_et)

                        meshpath = Path(
                            output_folder_vtk, f"{case}_{mesh_type[i]}_{num:03}.vtk"
                        )
                        write_vtk_surface(meshpath, vertices, faces_mapped)

                    # save closed RV mesh
                    output_folder_vtk = Path(output_folder, f"vtk{gp_suffix}")
                    output_folder_vtk.mkdir(exist_ok=True)

                    mesh_data = {"RV_septum": 1, "RV_freewall": 2, "tricuspid_valve": 6, "pulmonary_valve": 7}

                    combined_verts = np.array([]).reshape(0, 3)
                    combined_faces = np.array([], dtype=np.int64).reshape(0, 3)
                    offset = 0
                    for key in mesh_data:
                        start_fi = biventricular_model.surface_start_end[mesh_data[key]][0]
                        end_fi = biventricular_model.surface_start_end[mesh_data[key]][1] + 1
                        faces_et = biventricular_model.et_indices[start_fi:end_fi]

                        unique_inds = np.unique(faces_et.flatten())
                        vertices = biventricular_model.et_pos[unique_inds]

                        # remap faces/indices to 0-indexing
                        mapping = {old_index: new_index for new_index, old_index in enumerate(unique_inds)}
                        faces_mapped = np.vectorize(mapping.get)(faces_et)

                        meshpath = Path(
                            output_folder_vtk, f"{case}_{key}_{num:03}.vtk"
                        )
                        write_vtk_surface(meshpath, vertices, faces_mapped)

                        combined_verts = np.vstack((combined_verts, vertices))
                        combined_faces = np.vstack((combined_faces, faces_mapped + offset))

                        # print(f'offset now {offset}')
                        offset += len(vertices)  # TODO: fix logic....

                    # remove duplicate points (from chatGPT)
                    # Create a dictionary to map old indices to new indices after removing duplicates
                    index_map = {}
                    new_vertices = []
                    new_faces = []

                    # Iterate over vertices to remove duplicates and update index_map
                    for old_index, vertex in enumerate(combined_verts):
                        vertex_tuple = tuple(vertex)
                        if vertex_tuple not in index_map:
                            new_index = len(new_vertices)
                            new_vertices.append(vertex)
                            index_map[vertex_tuple] = new_index

                    # Update the faces array with the new indices
                    for face in combined_faces:
                        new_face = [index_map[tuple(combined_verts[old_index])] for old_index in face]
                        new_faces.append(new_face)

                    combined_verts_clean = np.array(new_vertices)
                    combined_faces_clean = np.array(new_faces)

                    meshpath = Path(
                        output_folder_vtk, f"{case}_RV_closed_{num:03}.vtk"
                    )
                    write_vtk_surface(meshpath, combined_verts_clean, combined_faces_clean)
                    if logger is not None:
                        logger.success(f"Model successfully saved to {meshpath}")

                progress.advance(task)
                # if you want to plot time series in html files uncomment the next line(s)
                # plot_timeseries(TimeSeries_step1, output_folder, 'TimeSeries_step1.html')
                # plot_timeseries(TimeSeries_step2, output_folder, 'TimeSeries_step2.html')

        #DoneFile = Path(os.path.join(output_folder, "Done.txt"))
        #DoneFile.touch(exist_ok=True)
        #logger.success(f"Models successfully saved in {args.output_dir}")

    except KeyboardInterrupt:
        raise KeyboardInterruptError()


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

    logger.info(f"Found {len(case_dirs)} cases to fit.")
    # start processing...
    start_time = time.time()

    if not (args.format.endswith('.obj') or args.format.endswith('.vtk')):
        raise argparse.ArgumentTypeError('argument format must be .obj or .vtk')

    for case in case_dirs:
        #print(f"Processing case: {os.path.basename(case)}")
        logger.info(f"Processing {os.path.basename(case)}")
        if not args.overwrite and os.path.exists(os.path.join(args.output_dir, os.path.basename(case))):
            print("Folder already exists for this case. Proceeding to next case")
            continue

        perform_fitting(case, out_dir=args.output_dir, gp_suffix=args.gp_suffix, si_suffix=args.si_suffix,
                        frames_to_fit=None, output_format = args.format, logger=logger)

    logger.info(f"Total cases processed: {len(case_dirs)}")
    logger.info(f"Total time: {time.time() - start_time}")
    logger.success(f"Done. Results are saved in {args.output_dir}")