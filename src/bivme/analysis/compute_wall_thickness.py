import argparse
import os, sys
import csv
import re
from pathlib import Path
import numpy as np
import pathlib

from bivme import MODEL_RESOURCE_DIR
from bivme.meshing.mesh import Mesh
from bivme.meshing.mesh_io import export_to_obj

from loguru import logger
from rich.progress import Progress
import fnmatch
import pyvista as pv
import matplotlib.pyplot as plt
from pyezzi import compute_thickness
from scipy.interpolate import NearestNDInterpolator
import nibabel as nib

# for printing while progress bar is progressing
console = None
debug = False

save_segmentation = True
def find_wall_thickness(case_name: str, model_file: os.PathLike, output_folder: os.PathLike, biv_model_folder: os.PathLike, precision: int) -> None:
    """
        # Authors: cm
        # Date: 09/24

    """

    # get the frame number
    frame_name = re.search(r'Frame_(\d+)\.txt', str(model_file), re.IGNORECASE)[1]

    # read GP file
    control_points = np.loadtxt(model_file, delimiter=',', skiprows=1, usecols=[0, 1, 2]).astype(float)

    subdivision_matrix_file = biv_model_folder / "subdivision_matrix.txt"
    assert subdivision_matrix_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {subdivision_matrix_file} file!"

    elements_file = biv_model_folder / 'ETIndicesSorted.txt'
    assert elements_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {elements_file} file!"

    material_file = biv_model_folder / 'ETIndicesMaterials.txt'
    assert material_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {material_file} file!"

    thru_wall_file = biv_model_folder / 'epi_to_septum_ETindices.txt'
    assert thru_wall_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {thru_wall_file} file!"

    if control_points.shape[0] > 0:
        subdivision_matrix = (np.loadtxt(subdivision_matrix_file)).astype(float)
        faces = np.loadtxt(elements_file).astype(int)-1
        mat = np.loadtxt(material_file, dtype='str')

        et_thru_wall = np.loadtxt(thru_wall_file, delimiter='\t').astype(int)-1

        ## convert labels to integer corresponding to the sorted list
        # of unique labels types
        unique_material = np.unique(mat[:,1])

        materials = np.zeros(mat.shape)
        for index, m in enumerate(unique_material):
            face_index = mat[:, 1] == m
            materials[face_index, 0] = mat[face_index, 0].astype(int)
            materials[face_index, 1] = [index] * np.sum(face_index)

        # add material for the new facets
        new_elem_mat = [list(range(materials.shape[0], materials.shape[0] + et_thru_wall.shape[0])),
                        [len(unique_material)] * len(et_thru_wall)]

        vertices = np.dot(subdivision_matrix, control_points)
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
        for name, part in zip(['lv_endo', 'lv_epi', 'rv_endo', 'rv_epi'], [lv_endo, lv_epi, rv_endo, rv_epi]):
            lv_faces = part.elements
            lv_faces = np.pad(lv_faces, ((0, 0), (1, 0)), 'constant', constant_values=3)
            lv_mesh = pv.PolyData(part.nodes, lv_faces)

            density = 1
            x_min, x_max, y_min, y_max, z_min, z_max = lv_mesh.bounds
            xx = np.arange(x_min, x_max, density)
            yy = np.arange(y_min, y_max, density)
            zz = np.arange(z_min, z_max, density)
            x, y, z = np.meshgrid(xx, yy, zz)

            # Create unstructured grid from the structured grid
            grid = pv.StructuredGrid(x, y, z)
            ugrid = pv.UnstructuredGrid(grid)

            # get part of the mesh within the mesh's bounding surface.
            print(f"Computing {name} mask")
            lv_selection = ugrid.select_enclosed_points(lv_mesh.extract_surface(),
                                                     tolerance=0.0,
                                                     check_surface=False)

            mask = lv_selection['SelectedPoints'].view(bool)
            mask = mask.reshape(x.shape, order='F')
            mask = (mask>0).astype(int)
            masks[name] = np.array(mask)

            if debug:
                fig, axes = plt.subplots(1, 3)
                axes[0].imshow(masks[name].sum(0))
                axes[0].set_title('Sum along X-axis')
                axes[1].imshow(masks[name].sum(1))
                axes[1].set_title('Sum along Y-axis')
                axes[2].imshow(masks[name].sum(2))
                axes[2].set_title('Sum along Z-axis')
                plt.show()

                pl = pv.Plotter(shape=(1, 1))
                lv_voxels = pv.voxelize(lv_mesh, density=density, check_surface=False)
                pl.add_mesh(lv_voxels, color='g', show_edges=True)
                pl.show()


        labeled_image_lv = 2*(masks['lv_epi'] - masks['lv_endo']) + masks['lv_endo']
        labeled_image_rv = 2*(masks['rv_epi'] - masks['rv_endo']) + masks['rv_endo']

        if save_segmentation:
            ni_img = nib.Nifti1Image((labeled_image_lv+labeled_image_rv).astype(np.int8), affine=np.eye(4))
            nib.save(ni_img, output_folder / f"labeled_image_lvrv_{case_name}_{frame_name}.nii")

        lv_thickness = compute_thickness(labeled_image_lv)
        rv_thickness = compute_thickness(labeled_image_rv)

        if save_segmentation:
            ni_img = nib.Nifti1Image(lv_thickness.astype(np.float32), affine=np.eye(4))
            nib.save(ni_img, output_folder / f"lv_thickness_{case_name}_{frame_name}.nii")

            ni_img = nib.Nifti1Image(rv_thickness.astype(np.float32), affine=np.eye(4))
            nib.save(ni_img, output_folder / f"rv_thickness_{case_name}_{frame_name}.nii")



if __name__ == "__main__":
    biv_resource_folder = MODEL_RESOURCE_DIR

    # parse command-line argument
    parser = argparse.ArgumentParser(description="LV & RV mass and volume calculation")
    parser.add_argument('-mdir', '--model_dir', type=Path, help='path to biv models')
    parser.add_argument('-o', '--output_folder', type=Path, help='output path', default="./")
    parser.add_argument("-b", '--biv_model_folder', default=biv_resource_folder,
                        help="folder containing subdivision matrices"
                                 f" (default: {biv_resource_folder})")
    parser.add_argument("-pat", '--patterns', default="*",
                        help="folder patterns to include (default '*')")
    parser.add_argument("-p", '--precision',  type=int, default=2,
                        help="Output precision")
    args = parser.parse_args()

    assert args.model_dir.exists(), \
        f"model_dir does not exist."

    assert args.output_folder.exists(), \
        f"output_path does not exist."

    folders = [p.name for p in Path(args.model_dir).glob(args.patterns) if os.path.isdir(p)]
    logger.info(f"Found {len(folders)} model folders.")

    # For pyezzi compatibility
    np.int = np.int_

    for i, folder in enumerate(folders):
        ## TODO: recursive param with walk() filtering
        rule = re.compile(fnmatch.translate("*model_frame*.txt"), re.IGNORECASE)
        models = [args.model_dir / folder / Path(name) for name in os.listdir(args.model_dir / folder) if rule.match(name)]
        models = sorted(models)
        print(str(args.model_dir / folder / Path('UKBB_88878_1144060')))
        logger.info(f"Processing {str(args.model_dir / folder)} ({i+1}/{len(folders)})")
        with Progress(transient=True) as progress:
            task = progress.add_task("Calculating wall thickness", total=len(models))
            console = progress

            for biv_model_file in models:
                find_wall_thickness(folder, biv_model_file, args.output_folder, biv_resource_folder, args.precision)
                progress.advance(task)

    logger.success(f"Done. Results are saved in {args.output_folder}")