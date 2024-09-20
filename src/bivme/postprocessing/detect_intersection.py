#from __future__ import annotations
import numpy as np
import pyvista as pv
import argparse
import os, sys
from pathlib import Path
import re
import fnmatch
from copy import deepcopy
from plotly.offline import plot
import plotly.graph_objs as go

from bivme.meshing.mesh_io import write_vtk_surface, export_to_obj
from bivme import MODEL_RESOURCE_DIR
from bivme.meshing.mesh import Mesh
from bivme.fitting.BiventricularModel import BiventricularModel
from loguru import logger
from rich.progress import Progress
import tomli
import shutil
from bivme.fitting.surface_enum import Surface
from bivme.fitting.surface_enum import ContourType
from bivme.fitting.GPDataSet import GPDataSet
import pandas as pd 

biv_model_folder = MODEL_RESOURCE_DIR
from bivme.fitting.diffeomorphic_fitting_utils import (
    solve_least_squares_problem,
    solve_convex_problem,
)

# mapping 
contour_map = {
   Surface.RV_FREEWALL : ContourType.SAX_RV_FREEWALL,
   Surface.RV_SEPTUM : ContourType.SAX_RV_SEPTUM,
   Surface.LV_ENDOCARDIAL : ContourType.SAX_LV_ENDOCARDIAL,
   Surface.RV_INSERT : ContourType.RV_INSERT,
   Surface.APEX : ContourType.APEX_POINT,
   Surface.MITRAL_VALVE : ContourType.MITRAL_VALVE,
   Surface.TRICUSPID_VALVE : ContourType.TRICUSPID_VALVE,
   Surface.AORTA_VALVE : ContourType.AORTA_VALVE,
   Surface.PULMONARY_VALVE : ContourType.PULMONARY_VALVE}
   # : ContourType.SAX_RV_EPICARDIAL
   # : ContourType.SAX_LV_EPICARDIAL

def fix_intersection(case_name: str, config: dict, model_file: os.PathLike, output_folder: os.PathLike, biv_model_folder: os.PathLike = MODEL_RESOURCE_DIR, output_format: str =".vtk", gp_suffix: str ="") -> None:
    """
        # Authors: cm
        # Date: 09/24
    """

    reference_biventricular_model = BiventricularModel(biv_model_folder, collision_detection = True)

    fitted_model = deepcopy(reference_biventricular_model)
    fitted_model.control_mesh = np.loadtxt(model_file, delimiter=',', skiprows=1, usecols=[0, 1, 2]).astype(float)
    fitted_model.update_control_mesh(fitted_model.control_mesh)
    current_collision = fitted_model.detect_collision()
    inter = current_collision.difference(fitted_model.reference_collision) 

    if bool(inter):
        
        logger.warning(f"Intersections detected for case {os.path.basename(os.path.normpath(model_file))}")        
        logger.info(f"Refitting of {str(case_name)}")

        # create a separate output folder for each patient
        output_folder = Path(output_folder) / os.path.basename(case_name)
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # initialise GP dataset from fitted model
        gp_dataset = GPDataSet()

        points = []
        slices = []
        contour_types = []
        weights = []
        for surface, contours in contour_map.items():
            start, end = fitted_model.get_surface_vertex_start_end_index(surface)
            for idx in range(start, end):
                points.append(fitted_model.et_pos[idx,:])
                slices.append(0)
                contour_types.append(contours)
                weights.append(1.0)

        gp_dataset.points_coordinates = np.array(points)
        gp_dataset.slice_number = np.array(slices)
        gp_dataset.contour_type = np.array(contour_types)
        gp_dataset.weights = np.array(weights)

        residuals = 0

        gp_dataset.apex = fitted_model.et_pos[fitted_model.APEX_INDEX,]
        gp_dataset.mitral_centroid = fitted_model.et_pos[fitted_model.get_surface_vertex_start_end_index(Surface.MITRAL_VALVE)[1],:]
        gp_dataset.tricuspid_centroid = fitted_model.et_pos[fitted_model.get_surface_vertex_start_end_index(Surface.TRICUSPID_VALVE)[1],:]

        reference_biventricular_model.update_pose_and_scale(gp_dataset)

        # Perform linear fit
        solve_least_squares_problem(reference_biventricular_model, config["fitting_weights"]["guide_points"], gp_dataset, logger, collision_detection=True, model_prior = fitted_model)
        
        ## Perform diffeomorphic fit
        residuals += solve_convex_problem(
            reference_biventricular_model,
            gp_dataset,
            config["fitting_weights"]["guide_points"],
            config["fitting_weights"]["convex_problem"],
            config["fitting_weights"]["transmural"],
            logger, 
            collision_detection=True, 
            model_prior = fitted_model
        )

        # save results in .txt format, one file for each frame
        model_data = {
            "x": reference_biventricular_model.control_mesh[:, 0],
            "y": reference_biventricular_model.control_mesh[:, 1],
            "z": reference_biventricular_model.control_mesh[:, 2],
            "Frame": [-1] * len(reference_biventricular_model.control_mesh[:, 2]),
        }

        model_data_frame = pd.DataFrame(data=model_data)

        with open(Path(output_folder) / model_file.name, "w") as file:
            file.write(
                model_data_frame.to_csv(
                    header=True, index=False, sep=",", lineterminator="\n"
                )
            )

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
                start_fi = reference_biventricular_model.surface_start_end[value[type]][0]
                end_fi = reference_biventricular_model.surface_start_end[value[type]][1] + 1
                faces_et = reference_biventricular_model.et_indices[start_fi:end_fi]
                unique_inds = np.unique(faces_et.flatten())
                vertices = np.vstack((vertices, reference_biventricular_model.et_pos[unique_inds]))

                # remap faces/indices to 0-indexing
                mapping = {old_index: new_index for new_index, old_index in enumerate(unique_inds)}
                faces_mapped = np.vstack((faces_mapped, np.vectorize(mapping.get)(faces_et) + offset))
                offset += len(reference_biventricular_model.et_pos[unique_inds])

            if output_format == ".vtk":
                output_folder_vtk = Path(output_folder, f"vtk{gp_suffix}")
                output_folder_vtk.mkdir(exist_ok=True)
                mesh_path = Path(
                    output_folder_vtk, f"{model_file.stem}_{key}.vtk"
                )
                write_vtk_surface(str(mesh_path), vertices, faces_mapped)
                logger.success(f"{model_file.stem}_{key}.vtk successfully saved to {output_folder_vtk}")

            elif output_format == ".obj":
                output_folder_obj = Path(output_folder, f"obj{gp_suffix}")
                output_folder_obj.mkdir(exist_ok=True)
                mesh_path = Path(
                    output_folder_obj, f"{model_file.stem}_{key}.obj"
                )
                export_to_obj(mesh_path, vertices, faces_mapped)
                logger.success(f"{model_file.stem}_{key}.obj successfully saved to {output_folder_obj}")
            else:
                logger.error('argument format must be .obj or .vtk')
                return -1

        ##TODO remove duplicated code here - not sure how yet
        if config["output"]["export_control_mesh"]:
            for key, value in control_mesh_meshes.items():
                vertices = np.array([]).reshape(0, 3)
                faces_mapped = np.array([], dtype=np.int64).reshape(0, 3)

                offset = 0
                for type in value:
                    start_fi = reference_biventricular_model.control_mesh_start_end[value[type]][0]
                    end_fi = reference_biventricular_model.control_mesh_start_end[value[type]][1] + 1
                    faces_et = reference_biventricular_model.et_indices_control_mesh[start_fi:end_fi]
                    unique_inds = np.unique(faces_et.flatten())
                    vertices = np.vstack((vertices, reference_biventricular_model.control_mesh[unique_inds]))

                    # remap faces/indices to 0-indexing
                    mapping = {old_index: new_index for new_index, old_index in enumerate(unique_inds)}
                    faces_mapped = np.vstack((faces_mapped, np.vectorize(mapping.get)(faces_et) + offset))
                    offset += len(reference_biventricular_model.control_mesh[unique_inds])

                if output_format == ".vtk":
                    output_folder_vtk = Path(output_folder, f"vtk{gp_suffix}")
                    output_folder_vtk.mkdir(exist_ok=True)
                    mesh_path = Path(
                        output_folder_vtk, f"{model_file.stem}_{key}_control_mesh.vtk"
                    )
                    write_vtk_surface(str(mesh_path), vertices, faces_mapped)
                    logger.success(f"{model_file.stem}_{key}_control_mesh.vtk successfully saved to {output_folder_vtk}")

                elif output_format == ".obj":
                    output_folder_obj = Path(output_folder, f"obj{gp_suffix}")
                    output_folder_obj.mkdir(exist_ok=True)
                    mesh_path = Path(
                        output_folder_obj, f"{model_file.stem}_{key}_control_mesh.obj"
                    )
                    export_to_obj(mesh_path, vertices, faces_mapped)
                    logger.success(f"{model_file.stem}_{key}_control_mesh.obj successfully saved to {output_folder_obj}")
                else:
                    logger.error('argument format must be .obj or .vtk')
                    return -1

        return residuals

    else:
        logger.success(f"No intersection detected for {case_name} - moving on")
        return


if __name__ == "__main__":
    biv_resource_folder = MODEL_RESOURCE_DIR

    parser = argparse.ArgumentParser(description="Removes intersection between free wall and septum if presents")
    parser.add_argument('-config', '--config_file', type=str,
                        help='Config file containing fitting parameters (the one used for fitting).')
    args = parser.parse_args()

    # Load config  - the config needs to be the same as the one used for fitting!
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


    if not config["output"]["show_logging"]:
        logger.remove()

    log_level = "DEBUG"
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"

    if not (config["output"]["mesh_format"].endswith('.obj') or config["output"]["mesh_format"].endswith('.vtk')):
        logger.error(f'argument mesh_format must be .obj or .vtk. {config["output"]["mesh_format"]} given.')
        sys.exit(0)

    for mesh in config["output"]["output_meshes"]:
        if mesh not in ["LV_ENDOCARDIAL", "RV_ENDOCARDIAL", "EPICARDIAL"]:
            logger.error(f'argument output_meshes invalid. {mesh} given. Allowed values are "LV_ENDOCARDIAL", "RV_ENDOCARDIAL", "EPICARDIAL"')
            sys.exit(0)

    # save config file to the output folder
    output_folder = Path(config["output"]["output_directory"]) / "corrected_models"
    output_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config_file, output_folder)

    case_list = os.listdir(config["output"]["output_directory"])
    folders = [Path(config["output"]["output_directory"], case).as_posix() for case in case_list]

    logger.info(f"Found {len(folders)} model folders.")

    try:
        for i, folder in enumerate(folders):
            if os.path.isdir(folder):
                rule = re.compile(fnmatch.translate("*model_frame*.txt"), re.IGNORECASE)
                models = [folder / Path(name) for name in os.listdir(Path(folder)) if rule.match(name)]
                models = sorted(models)

                logger.info(f"Processing {str(folder)} ({i+1}/{len(folders)})")
                with Progress(transient=True) as progress:
                    task = progress.add_task("Checking for intersection...", total=len(models))
                    console = progress

                    for biv_model_file in models:
                        fix_intersection(folder, config, biv_model_file, output_folder, biv_resource_folder, output_format = config["output"]["mesh_format"], gp_suffix=config["input"]["gp_suffix"])
                        progress.advance(task)

        logger.success(f"Done. Results are saved in {output_folder}")
    except KeyboardInterrupt:
        logger.info(f"Program interrupted by the user")
        sys.exit(0)