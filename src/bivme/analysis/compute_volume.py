"""
15/09/2022 - Laura Dal Toso
Based on A.M's scripts.
Script for the measurement of LV and LV mass and volume from biventricular models.

06/08/2024 - Charlene Mauger
* Fixed volume calculation
* Moved hardcoded variables to CLI arguments
* added logging and progress bar
"""

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
import scipy.io

# for printing while progress bar is progressing
console = None

def find_volume(case_name: str, model_file: os.PathLike, output_file: os.PathLike, biv_model_folder: os.PathLike, precision: int) -> None:
    """
        # Authors: ldt, cm
        # Date: 09/22, revised 08/24 by cm

        This function measures the mass and volume of LV and RV.
        #--------------------------------------------------------------
        Inputs: case_name = model case name
                model_file = fitted model (.txt), containing only data relative to one frame
                output_file = path to the output csv file
                biv_model_folder = path to the model folder - default: MODEL_RESOURCE_DIR
                precision - output precision for the volumes
        Output: None
    """

    # get the frame number
    frame_name = re.search(r'Frame_(\d+)\.txt', str(model_file), re.IGNORECASE)[1]

    # read GP file
    control_points = np.loadtxt(model_file, delimiter=',', skiprows=1, usecols=[0, 1, 2]).astype(float)

    # assign values to dict
    results_dict = {'case': case_name} | {
        k: '' for k in ['lv_vol', 'rv_vol', 'lv_epivol', 'rv_epivol', 'lv_mass', 'rv_mass']
    }

    subdivision_matrix_file = biv_model_folder / "subdivision_matrix_sparse.mat"
    assert subdivision_matrix_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {subdivision_matrix_file} file!"

    elements_file = biv_model_folder / 'ETIndicesSorted.txt'
    assert elements_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {elements_file} file!"

    material_file = biv_model_folder / 'ETIndicesMaterials.txt'
    assert material_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {material_file} file!"

    thru_wall_file = biv_model_folder / 'thru_wall_et_indices.txt'
    assert thru_wall_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {thru_wall_file} file!"

    if control_points.shape[0] > 0:
        subdivision_matrix = scipy.io.loadmat(subdivision_matrix_file)['S'].toarray()
        faces = np.loadtxt(elements_file).astype(int)-1
        mat = np.loadtxt(material_file, dtype='str')

        # A.M. :there is a gap between septum surface and the epicardial
        #   Which needs to be closed if the RV/LV epicardial volume is needed
        #   this gap can be closed by using the et_thru_wall facets
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
        rv_endo = model.get_mesh_component([6, 9, 10, 11], reindex_nodes=False)

        # switching the normal direction for the septum
        rv_endo.elements[rv_endo.materials == 10, :] = \
            np.array([rv_endo.elements[rv_endo.materials == 10, 0],
                      rv_endo.elements[rv_endo.materials == 10, 2],
                      rv_endo.elements[rv_endo.materials == 10, 1]]).T

        lv_epi = model.get_mesh_component([0, 1, 3, 4, 5, 10, 13], reindex_nodes=False)
        # switching the normal direction for the thru wall
        lv_epi.elements[lv_epi.materials == 13, :] = \
            np.array([lv_epi.elements[lv_epi.materials == 13, 0],
                      lv_epi.elements[lv_epi.materials == 13, 2],
                      lv_epi.elements[lv_epi.materials == 13, 1]]).T

        # switching the normal direction for the septum
        rv_epi = model.get_mesh_component([6, 7, 8, 10, 11, 12, 13], reindex_nodes=False)
        rv_epi.elements[rv_epi.materials == 10, :] = \
            np.array([rv_epi.elements[rv_epi.materials == 10, 0],
                      rv_epi.elements[rv_epi.materials == 10, 2],
                      rv_epi.elements[rv_epi.materials == 10, 1]]).T

        lv_endo_vol = lv_endo.get_volume()
        rv_endo_vol = rv_endo.get_volume()
        lv_epi_vol = lv_epi.get_volume()
        rv_epi_vol = rv_epi.get_volume()

        rv_mass = (rv_epi_vol - rv_endo_vol) * 1.05  # mass in grams
        lv_mass = (lv_epi_vol - lv_endo_vol) * 1.05

        # assign values to dict
        results_dict['lv_vol'] = round(lv_endo_vol, precision)
        results_dict['rv_vol'] = round(rv_endo_vol, precision)
        results_dict['lv_epivol'] = round(lv_epi_vol, precision)
        results_dict['rv_epivol'] = round(rv_epi_vol, precision)
        results_dict['lv_mass'] = round(lv_mass, precision)
        results_dict['rv_mass'] = round(rv_mass, precision)

    # append to the output_file
    with open(output_file, 'a', newline='') as file:
        # print out measurements in spreadsheet
        volume_writer = csv.writer(file)
        volume_writer.writerow([case_name, frame_name, results_dict['lv_vol'], results_dict['lv_mass'],
                         results_dict['rv_vol'], results_dict['rv_mass'],
                         results_dict['lv_epivol'], results_dict['rv_epivol']])

if __name__ == "__main__":
    biv_resource_folder = MODEL_RESOURCE_DIR

    # parse command-line argument
    parser = argparse.ArgumentParser(description="LV & RV mass and volume calculation")
    parser.add_argument('-mdir', '--model_dir', type=Path, help='path to biv models')
    parser.add_argument('-o', '--output_path', type=Path, help='output path', default="./")
    parser.add_argument("-b", '--biv_model_folder', default=biv_resource_folder,
                        help="folder containing subdivision matrices"
                                 f" (default: {biv_resource_folder})")
    parser.add_argument("-pat", '--patterns', default="*",
                        help="folder patterns to include (default '*')")
    parser.add_argument("-p", '--precision',  type=int, default=2,
                        help="Output precision")
    args = parser.parse_args()

    fieldnames = ['name', 'frame', 'lv_vol', 'lvm', 'rv_vol', 'rvm', 'lv_epivol', 'rv_epivol']

    assert args.model_dir.exists(), \
        f"model_dir does not exist."

    if not args.output_path.exists():
        args.output_path.mkdir(parents=True, exist_ok=True) 

    folders = [p.name for p in Path(args.model_dir).glob(args.patterns) if os.path.isdir(p)]
    logger.info(f"Found {len(folders)} model folders.")

    output_volume_file = args.output_path / 'lvrv_volumes.csv'
    with open(output_volume_file, 'w', newline='') as f:
        # create output file and write header
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        logger.info(f"Created {args.output_path} for the results.")

    for i, folder in enumerate(folders):
        ## TODO: recursive param with walk() filtering
        rule = re.compile(fnmatch.translate("*model_frame*.txt"), re.IGNORECASE)
        models = [args.model_dir / folder / Path(name) for name in os.listdir(args.model_dir / folder) if rule.match(name)]
        models = sorted(models)

        logger.info(f"Processing {str(args.model_dir / folder)} ({i+1}/{len(folders)})")
        with Progress(transient=True) as progress:
            task = progress.add_task("Computing volume", total=len(models))
            console = progress

            for biv_model_file in models:
                find_volume(folder, biv_model_file, output_volume_file, biv_resource_folder, args.precision)
                progress.advance(task)

    logger.success(f"Done. Results are saved in {output_volume_file}")