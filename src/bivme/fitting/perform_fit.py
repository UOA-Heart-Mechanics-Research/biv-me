import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from pathlib import Path
import re
import fnmatch
from loguru import logger

from bivme.fitting.GPDataSet import GPDataSet
from bivme.fitting.build_model_tools import create_ref_dataset, gp_rv_valve_generator, set_default_weights
from bivme.fitting.BiventricularModel import BiventricularModel
from bivme.fitting.diffeomorphic_fitting_utils import solve_least_squares_problem

from bivme.fitting.diffeomorphic_fitting_utils import solve_convex_fast
from bivme import MODEL_RESOURCE_DIR
from bivme.fitting.fitting_tools import add_valves, get_verts_faces, get_verts_faces_control
from bivme.fitting.surface_enum import ControlMesh, Surface
from bivme.meshing.mesh_io import export_mesh


def perform_fitting(folder: str,  config: dict, out_dir: str ="./results/", gp_suffix: str ="", si_suffix: str ="", workers: int = 1,
                    output_format: str =".vtk", my_logger: logger = logger) -> float:
    try:
        # Find slice info file
        filename_info = Path(folder) / f"SliceInfoFile.txt"
        if not filename_info.exists():
            my_logger.error(f"Cannot find {filename_info} file! Skipping this model")
            return -1

        # Extract the patient name from the folder name
        case_name = os.path.basename(os.path.normpath(folder))

        # Find all the guide point files in the folder
        rule = re.compile(fnmatch.translate(f"GPFile_*.txt"), re.IGNORECASE)
        time_frame = [Path(folder) / Path(name) for name in os.listdir(Path(folder)) if rule.match(name)]
        frame_names = [re.search(r'GPFile_*(\d+)\.txt', str(file), re.IGNORECASE)[1] for file in time_frame]
        frame_names = sorted(frame_names)
        frames_to_fit = sorted(np.unique(frame_names))  # if you want to fit all _frames#

        if len(frame_names) == 0:
            my_logger.error(f"Cannot find any GPFiles in {folder}! Skipping this case")
            return -1

        # Create results and models output folders
        output_folder = Path(out_dir) / case_name
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # Measure shift using key reference frame (e.g., ed_frame)
        start_time = time.time()
        my_logger.info(f"Creating reference dataset...")
        try:
            ed_dataset, shift_to_apply, updated_slice_position = create_ref_dataset(config,
                                                                                    folder,
                                                                                    output_folder,
                                                                                    filename_info,
                                                                                    case_name,
                                                                                    frame_names,
                                                                                    logger)
        except TypeError:
            my_logger.error(f"Cannot initialize reference dataset! Skipping this case")
            return -1
        
        # Create reference biv model and update its pose/scale
        aligned_biv_model = BiventricularModel(MODEL_RESOURCE_DIR, folder)
        aligned_biv_model.update_pose_and_scale(ed_dataset)
        logger.info(f"[CHECKPOINT][REF] Creating reference dataset took: {time.time() - start_time}s")

        # Prepare all guide point datasets upfront (to avoid read/write parallelization slowdowns)
        start_time = time.time()
        my_logger.info(f"Creating all guide point datasets upfront...")
        gp_dataset_list = prepare_all_gp_datasets(config, frames_to_fit, output_folder, folder, filename_info,
                                                  case_name, shift_to_apply, updated_slice_position, logger)
        logger.info(f"[CHECKPOINT][PREP] Preparing guide point models took: {time.time() - start_time}s")

        # Fit all frames in parallel
        start_time = time.time()
        my_logger.info(f"Fitting...")

        errors = {}
        model_dict = {}
        residual_dict = {}
        dataset_dict = {}
        total_residual = 0.0

        with ThreadPoolExecutor(max_workers=workers) as ex:
            # Set up futures queue
            futs = {
                ex.submit(_fit_one_frame, config, data_set, aligned_biv_model): data_set
                for data_set in gp_dataset_list
            }
            for idx,fut in enumerate(as_completed(futs)):
                data_set = futs[fut]
                if data_set is None:
                    num = f"{idx:.3f}" # get frame number from index
                    errors[num] = RuntimeError("Failed to prepare dataset")
                    continue
                num = data_set.get_frame_num()
                try:
                    biv_model, residual = fut.result()
                    model_dict[num] = biv_model
                    residual_dict[num] = residual
                    dataset_dict[num] = data_set

                    logger.info(f"[CHECKPOINT][BIV] Fit model for phase {num}")
                except Exception as e:
                    errors[num] = e

        # Handle any accumulated errors (only called at end)
        if errors:
            # Try to throw a summarized error
            msg = "Some frames failed: " + ", ".join(f"{k}: {type(v).__name__}" for k, v in errors.items())
            logger.error(msg)

        # Perform refits if necessary
        # Refitting works by checking if any frames have residuals above a certain threshold (e.g., 20% above median), and if so, 
        # we attempt to re-fit those frames starting from a smaller initial model (e.g., at 0.67 scale) which can help with convergence in some outlier frames. 
        if config["refitting"]["perform_refits"]:
            logger.info(f"[CHECKPOINT][REFIT] Checking for outlier frames to re-fit...")

            errors = {} # reset errors for refitting
            limit = config["refitting"]["threshold_factor"] # set limit X% above median to trigger refit
            scale_steps = [1, 0.67, 0.33]

            # Sort residual dict by largest residuals first 
            # This way we recalculate the median after refitting the most worst model each time and can potentially catch more outliers in one pass (if they exist)
            sorted_residual_dict = dict(sorted(residual_dict.items(), key=lambda item: item[1], reverse=True))

            for num, residual in sorted_residual_dict.items():
                if residual < 0: # skip frames that failed to fit at all
                    continue
                
                median_residual = np.median(list(residual_dict.values()))
                threshold = median_residual * limit
                
                if residual > threshold: # if residual is above threshold, attempt refit at different scales, starting from the second scale step (since the first is the original fit)
                    try:
                        model_dict[num], residual_dict[num] = refit_one_frame(config, dataset_dict[num], model_dict[num], residual, aligned_biv_model, scale_steps, threshold, num, logger)
                    except Exception as e:
                        errors[num] = e

            # Handle any accumulated errors (only called at end)
            if errors:
                # Try to throw a summarized error
                msg = "Some frames failed during refitting: " + ", ".join(f"{k}: {type(v).__name__}" for k, v in errors.items())
                logger.error(msg)

        # Log final residuals after refits
        for num, residual in residual_dict.items():
            logger.info(f"[CHECKPOINT][BIV] Fit model for phase {num}")
            logger.info(f"[CHECKPOINT][RES] Residual: {residual}")

        total_residual = sum(residual for residual in residual_dict.values() if residual >= 0) # only sum successful fits

        logger.info(f"[CHECKPOINT][FIT] Fitting models took: {time.time() - start_time}s")

        # Finalize
        start_time = time.time()

        my_logger.info(f"Writing out biventricular model data to files...")
        write_all_biv_models(config, model_dict, output_format, output_folder, case_name, logger)
        logger.info(f"[CHECKPOINT][WRITE] Writing out biv models to disk took: {time.time() - start_time}s")

        # Write GP files for plotting
        my_logger.info(f"Writing out guide point files...")
        write_all_gpfiles(gp_dataset_list, output_folder, case_name, logger)
        logger.info(f"[CHECKPOINT][WRITE] Writing out GP files to disk took: {time.time() - start_time}s")

        # Write residuals to a text file
        write_residuals_to_file(residual_dict, output_folder, logger)

        return total_residual / len(frames_to_fit) # return average residual across all frames (only counting successful fits)

    except KeyboardInterrupt:
        return -1

def prepare_all_gp_datasets(config,
                        frames_to_fit,
                        output_folder,
                        folder,
                        filename_info,
                        case_name,
                        shift_to_apply,
                        updated_slice_position,
                        logger):

    gp_dataset_list = []
    for frame_num in frames_to_fit:
        # Extract config parameters
        bh_corr_method = config["breathhold_correction"]["shifting"]
        gp_ds_factor = config["gp_processing"]["sampling"]

        # Create file
        # TODO: Move this to export function
        logger.info(f"Processing frame #{frame_num}")
        model_file = Path(output_folder, f"{case_name}_model_frame_{frame_num:03}.txt")
        model_file.touch(exist_ok=True)

        # Check if GP file exists
        filename = Path(folder) / f"GPFile_{frame_num:03}.txt"
        if not filename.exists():
            logger.error(f"Cannot find {filename} file! Skipping this model")
            gp_dataset_list.append(None)
            continue

        # Start processing this frame
        data_set = GPDataSet(str(filename), str(filename_info), case_name, sampling=gp_ds_factor, frame_num=frame_num)
        if not data_set.success:
            logger.error(f"Cannot initialize GPDataSet! Skipping this frame")
            gp_dataset_list.append(None)
            continue

        # Apply breath-hold correction if needed
        if bh_corr_method != "none":
            data_set.apply_slice_shift_fast(shift_to_apply, updated_slice_position)
            data_set.get_unintersected_slices_fast()

        # Generates RV epicardium and valve phantoms (if needed)
        # TODO: Make rv_thickness a config parameter
        gp_rv_valve_generator(data_set, config, logger, rv_thickness=3)
        set_default_weights(data_set)  # Example on how to set different weights for different points group (R.B.)

        gp_dataset_list.append(data_set)

    return gp_dataset_list

def _fit_one_frame(config, data_set, aligned_biv_model):
    if data_set is None:
        return None, -1

    # Get config parameters upfront
    gp_weight = config["fitting_weights"]["guide_points"]
    conv_weight = config["fitting_weights"]["convex_problem"]
    trans_weight = config["fitting_weights"]["transmural"]
    lsq_trans_weight = config["fitting_weights"]["lsq_trans_weight"]

    # Perform linear fit
    biv_model = aligned_biv_model.copy()
    solve_least_squares_problem(biv_model, gp_weight, data_set, lsq_trans_weight, collision_detection=False, model_prior=None, my_logger=logger)

    ## Perform diffeomorphic fit
    residual = solve_convex_fast(biv_model, data_set, gp_weight, conv_weight, trans_weight, collision_detection=False, model_prior=None, my_logger=logger)

    return biv_model, residual

def refit_one_frame(config, data_set, biv_model, residual, aligned_biv_model, scale_steps, threshold, num, logger):
    logger.warning(f"High residual ({residual:.2f}>{threshold:.2f}) detected for frame {num}, re-fitting at scale {scale_steps[1]}.")

    retry_successful = False
    for scale in scale_steps[1:]:
        scale_idx = scale_steps.index(scale)
        aligned_biv_model.update_pose_and_scale(data_set, scale) # update aligned model for this frame
        retry_biv_model, retry_residual = _fit_one_frame(config, data_set, aligned_biv_model)
        if retry_residual < residual:
            biv_model = retry_biv_model
            residual = retry_residual
            retry_successful = True

            if residual <= threshold:
                logger.info(f"[CHECKPOINT][REFIT] Re-fit successful with scale {scale}, updating model for frame {num}.")
                break
            else:
                if scale_idx + 1 < len(scale_steps):
                    logger.warning(f"Re-fit at scale {scale} improved residual but still high for frame {num}, trying again with scale {scale_steps[scale_idx+1]}.")
                else:
                    logger.warning(f"Re-fit at scale {scale} improved residual but still high for frame {num}.")
        else:
            if scale_idx + 1 < len(scale_steps):
                logger.warning(f"Re-fit at scale {scale} did not improve residual for frame {num}, trying again with scale {scale_steps[scale_idx+1]}.")
            else:
                logger.warning(f"Re-fit at scale {scale} did not improve residual for frame {num}.")

    if not retry_successful:
        logger.info(f"[CHECKPOINT][REFIT] All re-fits failed to improve residual for frame {num}, keeping original fit.")
    elif residual > threshold:
        logger.info(f"[CHECKPOINT][REFIT] Re-fits could not reduce residual below threshold for frame {num}, but we keep the improved fit.")

    return biv_model, residual

def write_all_biv_models(config, model_dict, output_format, output_folder, case_name, logger):
    # Get config parameters
    is_closed = config["output_fitting"]["closed_mesh"]
    meshes_to_export = config["output_fitting"]["output_meshes"]
    export_control_mesh = config["output_fitting"]["export_control_mesh"]

    # Write out models as text files
    for phase, biv_model in model_dict.items():
        arr = np.column_stack([
            biv_model.control_mesh[:, 0],  # x
            biv_model.control_mesh[:, 1],  # y
            biv_model.control_mesh[:, 2],  # z
            np.full(len(biv_model.control_mesh), phase)  # Frame
        ])
        header = "x,y,z,Frame"
        model_file = Path(output_folder, f"{case_name}_model_frame_{phase:03}.txt")
        np.savetxt(model_file, arr, delimiter=",", header=header, comments="", fmt="%s,%s,%s,%s")

        # Save VTK or OBJ files
        if output_format != "none":
            meshes = {}
            for surface in Surface:
                if surface.name in meshes_to_export:
                    mesh_data = {surface.name: surface.value}
                    add_valves(surface.name, mesh_data, is_closed) # add valves if want closed
                    meshes[surface.name] = mesh_data

            # Special handling for RV endo
            if "RV_ENDOCARDIAL" in meshes_to_export:
                rv_mesh = {"RV_SEPTUM": Surface.RV_SEPTUM.value, "RV_FREEWALL": Surface.RV_FREEWALL.value}
                add_valves("RV_ENDOCARDIAL", rv_mesh, is_closed)
                meshes["RV_ENDOCARDIAL"] = rv_mesh

            # Export all requested meshes
            for key, value in meshes.items():
                vertices, faces_mapped = get_verts_faces(biv_model, value)
                mesh_filename = f"{case_name}_{key}_{phase:03}{output_format}"
                export_mesh(output_format, output_folder, mesh_filename, vertices, faces_mapped, logger)

            # Export control mesh (if applicable)
            if export_control_mesh:
                control_mesh_meshes = {}
                for surface in ControlMesh:
                    if surface.name in meshes_to_export:
                        control_mesh_mesh_data = {surface.name: surface.value}
                        control_mesh_meshes[surface.name] = control_mesh_mesh_data

                for key, value in control_mesh_meshes.items():
                    vertices, faces_mapped = get_verts_faces_control(biv_model, value)
                    mesh_filename = f"{case_name}_{key}_{phase:03}_control_mesh{output_format}"
                    export_mesh(output_format, output_folder, mesh_filename, vertices, faces_mapped, logger)

def write_all_gpfiles(gpdataset_list, output_folder, case_name, logger):
    mdata_saved = False
    for data_set in gpdataset_list:
        if data_set is None:
            continue

        gpfile_folder = Path(output_folder) / "gpfiles"
        gpfile_folder.mkdir(parents=True, exist_ok=True)

        # Write slice info file (only once)
        if not mdata_saved:
            metadata_filename = gpfile_folder / f"{case_name}_SliceInfoFile.txt"
            data_set.write_sliceinfofile(str(metadata_filename))
            logger.success(f"{os.path.basename(metadata_filename)} successfully saved to {metadata_filename}")
            mdata_saved = True

        # Write GP file
        frame_num = data_set.get_frame_num()
        gpfile_name = gpfile_folder / f"{case_name}_GPFile_{frame_num:03}.txt"
        data_set.write_gpfile(str(gpfile_name), int(frame_num), overwrite=True)
        logger.success(f"{os.path.basename(gpfile_name)} successfully saved to {gpfile_name}")

def write_residuals_to_file(residual_dict, output_folder, logger):
    residuals_filename = Path(output_folder) / f"residuals.txt"
    with open(residuals_filename, 'w') as f:
        f.write("Frame,Residual\n")
        for frame, residual in sorted(residual_dict.items()):
            f.write(f"{frame},{residual}\n")
    logger.success(f"Residuals successfully saved to {residuals_filename}")