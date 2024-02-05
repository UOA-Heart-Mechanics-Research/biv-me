import os
import numpy as np
import time
import pandas as pd
import pyvista as pv
import plotly.graph_objs as go
from pathlib import Path
from plotly.offline import plot

from bivme.fitting.BiventricularModel import BiventricularModel
from bivme.fitting.GPDataSet import GPDataSet
from bivme.fitting.surface_enum import ContourType
from bivme.fitting.Diffeomorphic_fitting import (
    MultiThreadSmoothingED,
    SolveProblemCVXOPT,
    plot_timeseries,
)

from bivme.fitting.config_params import *

# This list of contours_to _plot was taken from Liandong Lee
contours_to_plot = [
    ContourType.LAX_RA,
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
    ContourType.SAX_RV_EPICARDIAL,
    ContourType.LAX_RV_EPICARDIAL,
    ContourType.LAX_LV_ENDOCARDIAL,
    ContourType.LAX_LV_EPICARDIAL,
    ContourType.LAX_RV_EPICARDIAL,
    ContourType.SAX_RV_OUTLET,
    ContourType.AORTA_PHANTOM,
    ContourType.TRICUSPID_PHANTOM,
    ContourType.MITRAL_PHANTOM,
]


def write_vtk_surface(filename, vertices, faces):
    """
    Write a VTK surface mesh.

    Parameters
    ----------
    filename : str
        The name of the output VTK file.
    vertices : numpy.ndarray
        An array of shape (N, 3) representing the vertex coordinates.
    faces : numpy.ndarray
        An array of shape (M, 3) representing the triangular faces.

    Returns
    -------
    None
    """

    mesh = pv.PolyData(vertices, np.c_[np.ones(len(faces)) * 3, faces].astype(int))
    mesh.save(filename, binary=False)


def perform_fitting(folder, outdir="./results/", gp_suffix="", si_suffix="", **kwargs):
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

        filename = os.path.join(folder, f"GPFile{gp_suffix}.txt")
        filenameInfo = os.path.join(folder, f"SliceInfoFile{si_suffix}.txt")

        # extract the patient name from the folder name
        case = os.path.basename(os.path.normpath(folder))
        print("case: ", case)

        # read all the frames from the GPFile
        all_frames = pd.read_csv(filename, sep="\t")
        # select which frames to fit
        ED_frame = None
        try:
            ED_frame = int(case_frame_dict[str(case)][0])
        except:
            ED_frame = 0
            print("ED set to frame # 0")

        # frames_to_fit = np.array(case_frame_dict[str(case)]) # oly fit ED and ES, if ED_ES file provided
        frames_to_fit = np.unique(
            [i[6] for i in all_frames.values]
        )  # if you want to fit all _frames

        # path to model template files
        model_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "../../model"
        )

        # create a separate output folder for each patient
        output_folder = os.path.join(outdir, case)
        try:
            os.makedirs(output_folder, exist_ok=True)
        except:
            raise ValueError

        # create log Files where to store fitting errors and shift
        Errorfile = Path(os.path.join(output_folder, f"Errorfile{gp_suffix}.txt"))
        Errorfile.touch(exist_ok=True)
        Shiftfile = Path(os.path.join(output_folder, f"Shiftfile{gp_suffix}.txt"))
        Shiftfile.touch(exist_ok=True)
        Posfile = Path(os.path.join(output_folder, f"Posfile{gp_suffix}.txt"))
        Posfile.touch(exist_ok=True)

        with open(Errorfile, "w") as f:
            f.write("Log for patient: " + case + "\n")

        # The next lines are used to measure shift using only a key frame
        if measure_shift_EDonly == True:
            print("shift measured only at ED frame")

            ED_dataset = GPDataSet(
                filename,
                filenameInfo,
                case,
                sampling=sampling,
                time_frame_number=ED_frame,
            )
            result_ED = ED_dataset.sinclaire_slice_shifting(frame_num=ED_frame)
            shift_ED = result_ED[0]
            pos_ED = result_ED[1]
            # np.save(os.path.join(output_folder, 'shift.txt'), shift_ED)
            with open(Shiftfile, "w") as file:
                file.write("shift measured only at ED: frame " + str(ED_frame) + "\n")
                file.write(str(shift_ED))
                file.close()

            with open(Posfile, "w") as file:
                file.write("pos measured only at ED: frame " + str(ED_frame) + "\n")
                file.write(str(pos_ED))
                file.close()

        # initialise time series lists
        TimeSeries_step1 = []
        TimeSeries_step2 = []

        print("FITTING OF ", str(case), "----> started \n")

        for idx, num in enumerate(sorted(frames_to_fit)):
            num = int(num)  # frame number
            print("frame num", num)

            Modelfile = Path(
                output_folder, f"{case}{gp_suffix}_model_frame_{num:03}.txt"
            )
            Modelfile.touch(exist_ok=True)

            with open(Errorfile, "a") as f:
                f.write("\nFRAME #" + str(int(num)) + "\n")

            data_set = GPDataSet(
                filename, filenameInfo, case, sampling=sampling, time_frame_number=num
            )
            biventricular_model = BiventricularModel(model_path, case)
            model = biventricular_model.plot_surface(
                "rgb(0,127,0)", "rgb(0,0,127)", "rgb(127,0,0)", surface="all"
            )

            if measure_shift_EDonly == True:
                # apply shift measured previously using ED frame
                data_set.apply_slice_shift(shift_ED, pos_ED)
                data_set.get_unintersected_slices()
            else:
                # measure and apply shift to current frame
                shiftedSlice = data_set.sinclaire_slice_shifting(Errorfile, int(num))
                shiftmeasure = shiftedSlice[0]
                posmeasure = shiftedSlice[1]

                if idx == 0:
                    with open(Shiftfile, "w") as file:
                        file.write("Frame number:  " + str(num) + "\n")
                        file.write(str(shiftmeasure))
                        file.close()

                    with open(Posfile, "w") as file:
                        file.write("Frame number:  " + str(num) + "\n")
                        file.write(str(posmeasure))
                        file.close()

                else:
                    with open(Shiftfile, "a") as file:
                        file.write("\nFrame number:  " + str(num) + "\n")
                        file.write(str(shiftmeasure))
                        file.close()

                    with open(Posfile, "w") as file:
                        file.write("\nFrame number:  " + str(num) + "\n")
                        file.write(str(posmeasure))
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

            mitral_points = data_set.create_valve_phantom_points(
                30, ContourType.MITRAL_VALVE
            )
            tri_points = data_set.create_valve_phantom_points(
                30, ContourType.TRICUSPID_VALVE
            )
            pulmonary_points = data_set.create_valve_phantom_points(
                20, ContourType.PULMONARY_VALVE
            )
            aorta_points = data_set.create_valve_phantom_points(
                20, ContourType.AORTA_VALVE
            )

            contourPlots = data_set.PlotDataSet(contours_to_plot)

            # Example on how to set different weights for different points group (R.B.)
            data_set.weights[data_set.contour_type == ContourType.MITRAL_PHANTOM] = 2
            data_set.weights[data_set.contour_type == ContourType.AORTA_PHANTOM] = 2
            data_set.weights[data_set.contour_type == ContourType.PULMONARY_PHANTOM] = 2
            data_set.weights[data_set.contour_type == ContourType.TRICUSPID_PHANTOM] = 2

            data_set.weights[data_set.contour_type == ContourType.APEX_POINT] = 1
            data_set.weights[data_set.contour_type == ContourType.RV_INSERT] = 5

            data_set.weights[data_set.contour_type == ContourType.MITRAL_VALVE] = 2
            data_set.weights[data_set.contour_type == ContourType.AORTA_VALVE] = 2
            data_set.weights[data_set.contour_type == ContourType.PULMONARY_VALVE] = 2

            # Perform linear fit (step1)
            MultiThreadSmoothingED(biventricular_model, weight_GP, data_set, Errorfile)

            # Plot results
            model = biventricular_model.plot_surface(
                "rgb(0,127,0)", "rgb(0,0,127)", "rgb(127,0,0)", "all"
            )
            data = model + contourPlots
            # TimeSeries_step1.append([data, num])

            # Perform diffeomorphic fit (step2)
            SolveProblemCVXOPT(
                biventricular_model,
                data_set,
                weight_GP,
                low_smoothing_weight,
                transmural_weight,
                Errorfile,
            )

            # Plot final results
            model = biventricular_model.plot_surface(
                "rgb(0,127,0)", "rgb(0,0,127)", "rgb(127,0,0)", "all"
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
                        header=True, index=False, sep=",", line_terminator="\n"
                    )
                )

            # save surface meshes as vtk
            output_folder_vtk = Path(output_folder, f"vtk{gp_suffix}")
            output_folder_vtk.mkdir(exist_ok=True)
            mesh_type = ["epicardium", "LV_endocardium", "RV_freewall", "RV_septum"]
            for i in range(4):
                vertices = np.vstack((data[i].x, data[i].y, data[i].z)).transpose()
                faces = np.vstack((data[i].i, data[i].j, data[i].k)).transpose()
                meshpath = Path(
                    output_folder_vtk, f"{case}_{mesh_type[i]}_{num:03}.vtk"
                )
                write_vtk_surface(meshpath, vertices, faces)
                
            # save closed RV mesh
            output_folder_vtk = Path(output_folder, f"vtk{gp_suffix}")
            output_folder_vtk.mkdir(exist_ok=True)
            
            mesh_data = {"RV_septum": 1, "RV_freewall": 2, "tricuspid_valve": 6, "pulmonary_valve": 7}
            
            combined_verts = np.array([]).reshape(0,3)
            combined_faces = np.array([], dtype=np.int64).reshape(0,3)
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
                
                print(f'offset now {offset}')
                offset += len(vertices) # TODO: fix logic.... 
            
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
            
        # if you want to plot time series in html files uncomment the next line(s)
        # plot_timeseries(TimeSeries_step1, output_folder, 'TimeSeries_step1.html')
        # plot_timeseries(TimeSeries_step2, output_folder, 'TimeSeries_step2.html')

        DoneFile = Path(os.path.join(output_folder, "Done.txt"))
        DoneFile.touch(exist_ok=True)

    except KeyboardInterrupt:
        raise KeyboardInterruptError()


if __name__ == "__main__":
    
    # directory containing guidepoint files
    dir_gp = r"R:\resmed201900006-biomechanics-in-heart-disease\Sandboxes\Debbie\collaborations\chicago-rv-mesh\analysis\gpfiles"
    dir_out = r"R:\resmed201900006-biomechanics-in-heart-disease\Sandboxes\Debbie\collaborations\chicago-rv-mesh\analysis\fitted"

    # set list of cases to process
    caselist = ["RV01", "RV02", "RV03", "RV04"]
    casedirs = [Path(dir_gp, case).as_posix() for case in caselist]

    # set guidepoint and slice info files to use
    gp_suffix = "_cim"
    si_suffix = "_cim"

    # start processing...
    starttime = time.time()

    [
        perform_fitting(case, outdir=dir_out, gp_suffix=gp_suffix, si_suffix=si_suffix)
        for case in casedirs
    ]

    print("TOTAL CASES:", len(casedirs))
    print("TOTAL TIME: ", time.time() - starttime)
