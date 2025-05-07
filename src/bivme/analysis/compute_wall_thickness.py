import argparse
import os, sys
import csv
import re
from pathlib import Path
import numpy as np
import pathlib

from bivme import MODEL_RESOURCE_DIR
from bivme.meshing.mesh import Mesh
from bivme.meshing.mesh_io import write_colored_vtk_surface

from loguru import logger
from rich.progress import Progress
import fnmatch
import pyvista as pv
from pyezzi.pyezzi.thickness import compute_thickness
import nibabel as nib
from bivme.fitting.BiventricularModel import BiventricularModel
import math
from scipy.spatial import cKDTree
from bivme.fitting.surface_enum import Surface
import scipy.io

# for printing while progress bar is progressing
console = None
debug = False

if debug:
    import matplotlib.pyplot as plt

def find_wall_thickness(case_name: str, model_file: os.PathLike, output_folder: os.PathLike, biv_model_folder: os.PathLike = MODEL_RESOURCE_DIR, voxel_resolution: float = 1.0, save_segmentation: bool = True) -> None:
    """
        # Authors: cm
        # Date: 09/24
    """

    # get the frame number
    frame_name = re.search(r'Frame_(\d+)\.txt', str(model_file), re.IGNORECASE)[1]

    biventricular_model = BiventricularModel(biv_model_folder)

    # read GP file
    biventricular_model.control_mesh = np.loadtxt(model_file, delimiter=',', skiprows=1, usecols=[0, 1, 2]).astype(float)

    material_file = biv_model_folder / 'ETIndicesMaterials.txt'
    assert material_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {material_file} file!"

    thru_wall_file = biv_model_folder / 'thru_wall_et_indices.txt'
    assert thru_wall_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {thru_wall_file} file!"

    if biventricular_model.control_mesh.shape[0] > 0:
        faces = biventricular_model.et_indices
        mat = np.loadtxt(material_file, dtype='str')
        et_thru_wall = np.loadtxt(thru_wall_file, delimiter='\t').astype(int)-1

        ## convert labels to integer corresponding to the sorted list of unique labels types
        unique_material = np.unique(mat[:,1])

        materials = np.zeros(mat.shape)
        for index, m in enumerate(unique_material):
            face_index = mat[:, 1] == m
            materials[face_index, 0] = mat[face_index, 0].astype(int)
            materials[face_index, 1] = [index] * np.sum(face_index)

        # add material for the new facets
        new_elem_mat = [list(range(materials.shape[0], materials.shape[0] + et_thru_wall.shape[0])),
                        [len(unique_material)] * len(et_thru_wall)]

        vertices = np.dot(biventricular_model.matrix, biventricular_model.control_mesh)
        faces = np.concatenate((faces.astype(int), et_thru_wall))
        materials = np.concatenate((materials.T, new_elem_mat), axis=1).T.astype(int)

        model = Mesh('mesh')
        model.set_nodes(vertices)
        model.set_elements(faces)
        model.set_materials(materials[:, 0], materials[:, 1])

        # components list, used to get the correct mesh components:
        # ['0 AORTA_VALVE' '1 AORTA_VALVE_CUT' '2 LV_ENDOCARDIAL' '3 LV_EPICARDIAL'
        # ' 4 MITRAL_VALVE' '5 MITRAL_VALVE_CUT' '6 PULMONARY_VALVE' '7 PULMONARY_VALVE_CUT'
        # '8 RV_EPICARDIAL' '9 RV_FREEWALL' '10 RV_SEPTUM' '11 TRICUSPID_VALVE'
        # '12 TRICUSPID_VALVE_CUT', '13' THRU WALL]

        lv_endo = model.get_mesh_component([0, 2, 4], reindex_nodes=False)

        # Select RV endocardial
        lv_epi = model.get_mesh_component([0, 1, 3, 4, 5, 10, 13], reindex_nodes=False)
        # switching the normal direction for the thru wall

        rv_endo = model.get_mesh_component([6, 9, 10, 11], reindex_nodes=False)
        # switching the normal direction for the septum

        # switching the normal direction for the septum
        rv_epi = model.get_mesh_component([6, 7, 8, 10, 11, 12, 13], reindex_nodes=False)
        masks = {}

        model_x_min = model_y_min = model_z_min = 1000
        model_x_max = model_y_max = model_z_max = -1000
        for name, part in zip(['lv_epi', 'rv_epi'], [lv_epi, rv_epi]):
            lv_faces = part.elements
            lv_faces = np.pad(lv_faces, ((0, 0), (1, 0)), 'constant', constant_values=3)
            lv_mesh = pv.PolyData(part.nodes, lv_faces)
            x_min, x_max, y_min, y_max, z_min, z_max = lv_mesh.bounds
            model_x_min = min(model_x_min, x_min)
            model_y_min = min(model_y_min, y_min)
            model_z_min = min(model_z_min, z_min)
            model_x_max = max(model_x_max, x_max)
            model_y_max = max(model_y_max, y_max)
            model_z_max = max(model_z_max, z_max)

        ##TODO do we want to keep the same bound for ED and ES - does that change the outcome?
        xx = np.arange(model_x_min, model_x_max, voxel_resolution)
        yy = np.arange(model_y_min, model_y_max, voxel_resolution)
        zz = np.arange(model_z_min, model_z_max, voxel_resolution)
        x, y, z = np.meshgrid(xx, yy, zz)

        # Create unstructured grid from the structured grid
        grid = pv.StructuredGrid(x, y, z)
        ugrid = pv.UnstructuredGrid(grid)

        for name, part in zip(['lv_endo', 'lv_epi', 'rv_endo', 'rv_epi'], [lv_endo, lv_epi, rv_endo, rv_epi]):
            lv_faces = part.elements
            lv_faces = np.pad(lv_faces, ((0, 0), (1, 0)), 'constant', constant_values=3)
            lv_mesh = pv.PolyData(part.nodes, lv_faces)

            # get part of the mesh within the mesh's bounding surface.
            logger.info(f"Voxelising {name}")
            surface = lv_mesh.extract_surface()
            lv_selection = ugrid.select_enclosed_points(surface,
                                                     tolerance=0.0,
                                                     check_surface=False)

            mask = lv_selection['SelectedPoints'].view(bool)
            mask = mask.reshape(x.shape, order='F')
            mask = (mask>0).astype(int)
            masks[name] = np.array(mask)

            if debug:
                pl = pv.Plotter(shape=(1, 1))
                lv_voxels = pv.voxelize(lv_mesh, density=voxel_resolution, check_surface=False)
                pl.add_mesh(lv_voxels, color='g', show_edges=True)
                pl.show()

        labeled_image_lv = 2*(masks['lv_epi'] - masks['lv_endo']) + masks['lv_endo']
        labeled_image_rv = 2*(masks['rv_epi'] - masks['rv_endo']) + masks['rv_endo']

        if save_segmentation:
            logger.info(f'Saving wall thickness image to {Path(output_folder / f"labeled_image_lv_{case_name}_{frame_name}.nii")}')
            ni_img = nib.Nifti1Image(labeled_image_lv.astype(np.int8), affine=np.eye(4))
            nib.save(ni_img, output_folder / f"labeled_image_lv_{case_name}_{frame_name}.nii")

            logger.info(f'Saving wall thickness image to {Path(output_folder / f"labeled_image_rv_{case_name}_{frame_name}.nii")}')
            ni_img = nib.Nifti1Image(labeled_image_rv.astype(np.int8), affine=np.eye(4))
            nib.save(ni_img, output_folder / f"labeled_image_rv_{case_name}_{frame_name}.nii")

        lv_thickness = compute_thickness(labeled_image_lv)
        rv_thickness = compute_thickness(labeled_image_rv)

        if save_segmentation:
            ni_img = nib.Nifti1Image(lv_thickness.astype(np.float32), affine=np.eye(4))
            nib.save(ni_img, output_folder / f"lv_thickness_{case_name}_{frame_name}.nii")
            logger.info(
                f'Saving wall thickness image to {Path(output_folder / f"lv_thickness_{case_name}_{frame_name}.nii")}')

            ni_img = nib.Nifti1Image(rv_thickness.astype(np.float32), affine=np.eye(4))
            nib.save(ni_img, output_folder / f"rv_thickness_{case_name}_{frame_name}.nii")
            logger.info(
                f'Saving wall thickness image to {Path(output_folder / f"rv_thickness_{case_name}_{frame_name}.nii")}')

        labeled_image_lv = labeled_image_lv.reshape(ugrid.n_points, order='F')
        labeled_image_rv = labeled_image_rv.reshape(ugrid.n_points, order='F')
        lv_thickness = lv_thickness.reshape(ugrid.n_points, order='F')
        rv_thickness = rv_thickness.reshape(ugrid.n_points, order='F')

        idx_to_keep = np.where(labeled_image_lv == 2)[0] # only keep voxels belonging to the wall for fast processing
        lv_thickness = lv_thickness[idx_to_keep]
        lv_tree = cKDTree(ugrid.points[idx_to_keep,:])

        idx_to_keep = np.where(labeled_image_rv == 2)[0] # only keep voxels belonging to the wall for fast processing
        rv_thickness = rv_thickness[idx_to_keep]
        rv_tree = cKDTree(ugrid.points[idx_to_keep, :])

        meshes = {}
        mesh_data = {}
        mesh_data["LV_ENDOCARDIAL"] = Surface.LV_ENDOCARDIAL.value
        meshes["LV_ENDOCARDIAL"] = mesh_data

        mesh_data = {}
        mesh_data["RV_SEPTUM"] = Surface.RV_SEPTUM.value
        mesh_data["RV_FREEWALL"] = Surface.RV_FREEWALL.value
        meshes["RV_ENDOCARDIAL"] = mesh_data

        for key, value in meshes.items():
            mesh_vertices = np.array([]).reshape(0, 3)
            faces_mapped = np.array([], dtype=np.int64).reshape(0, 3)

            if key == 'LV_ENDOCARDIAL':
               tree = lv_tree
               wall_thickness = lv_thickness
            elif key == 'RV_ENDOCARDIAL':
               tree = rv_tree
               wall_thickness = rv_thickness
            else:
                continue

            thickness = []
            offset = 0
            for type in value:
                start_fi = biventricular_model.surface_start_end[value[type]][0]
                end_fi = biventricular_model.surface_start_end[value[type]][1] + 1
                faces_et = biventricular_model.et_indices[start_fi:end_fi]
                unique_inds = np.unique(faces_et.flatten())
                mesh_vertices = np.vstack((mesh_vertices, vertices[unique_inds]))

                for vertex in vertices[unique_inds]:
                    _, indx = tree.query(vertex, k=1, p=2)
                    w_t = wall_thickness[indx]
                    if math.isnan(w_t):
                        w_t = 0
                    thickness.append(w_t)

                # remap faces/indices to 0-indexing
                mapping = {old_index: new_index for new_index, old_index in enumerate(unique_inds)}
                faces_mapped = np.vstack((faces_mapped, np.vectorize(mapping.get)(faces_et) + offset))
                offset += len(biventricular_model.et_pos[unique_inds])

            output_vtk = Path(output_folder) / f'{case_name}_{frame_name:03}_wall_thickness_{key}.vtk'
            write_colored_vtk_surface(output_vtk, mesh_vertices, faces_mapped,
                                      np.array([thickness, [0]*len(thickness), [0]*len(thickness)]).T)

            logger.success(f"{case_name}_{frame_name:03}.vtk successfully saved to {output_folder}")


if __name__ == "__main__":
    biv_resource_folder = MODEL_RESOURCE_DIR

    # parse command-line argument
    parser = argparse.ArgumentParser(description="Wall thickness calculation from 3D masks")
    parser.add_argument('-mdir', '--model_dir', type=Path, help='path to biv models')
    parser.add_argument('-o', '--output_folder', type=Path, help='output path', default="./")
    parser.add_argument("-b", '--biv_model_folder', default=biv_resource_folder,
                        help="folder containing subdivision matrices"
                                 f" (default: {biv_resource_folder})")
    parser.add_argument("-pat", '--patterns', default="*",
                        help="folder patterns to include (default '*')")
    parser.add_argument("-r", '--voxel_resolution',  type=float, default=1,
                        help="Output precision")
    parser.add_argument("-s", '--save_segmentation', action="store_true",
                        help="Boolean value indicating if we want the 3D masks to be saved")
    args = parser.parse_args()

    assert args.model_dir.exists(), \
        f"model_dir does not exist."

    if not args.output_folder.exists():
        args.output_folder.mkdir(parents=True, exist_ok=True) 

    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True)

    folders = [p.name for p in Path(args.model_dir).glob(args.patterns) if os.path.isdir(p)]
    logger.info(f"Found {len(folders)} model folders.")

    try:
        for i, folder in enumerate(folders):
            ## TODO: recursive param with walk() filtering
            rule = re.compile(fnmatch.translate("*model_frame*.txt"), re.IGNORECASE)
            models = [args.model_dir / folder / Path(name) for name in os.listdir(args.model_dir / folder) if rule.match(name)]
            models = sorted(models)

            participant_output = Path(output_folder) / folder
            participant_output.mkdir(exist_ok=True)

            logger.info(f"Processing {str(args.model_dir / folder)} ({i+1}/{len(folders)})")
            with Progress(transient=True) as progress:
                task = progress.add_task("Calculating wall thickness", total=len(models))
                console = progress

                for biv_model_file in models:
                    find_wall_thickness(folder, biv_model_file, participant_output, biv_resource_folder, args.voxel_resolution, args.save_segmentation)
                    progress.advance(task)

        logger.success(f"Done. Results are saved in {output_folder}")
    except KeyboardInterrupt:
        logger.info(f"Program interrupted by the user")
        sys.exit(0)