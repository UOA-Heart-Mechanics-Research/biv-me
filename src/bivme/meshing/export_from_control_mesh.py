import os
import time
import sys
import argparse

import numpy as np
from pathlib import Path
import re
from loguru import logger

from bivme.fitting.BiventricularModel import BiventricularModel

from bivme import MODEL_RESOURCE_DIR
from bivme.fitting.fitting_tools import add_valves, get_verts_faces
from bivme.fitting.surface_enum import Surface
from bivme.meshing.mesh_io import export_mesh


def export_from_control_mesh(model_folder: Path, output_directory: Path, closed_mesh: bool, surfaces_to_export: list, output_format: str = '.vtk', case_name: str = None, logger: logger = logger):
    # Get all model files in the folder
    model_files = list(model_folder.glob(f"*model_frame_*.txt"))

    for model_file in model_files:
        # get the frame number
        frame_name = re.search(r'Frame_(\d+)\.txt', str(model_file), re.IGNORECASE)[1]

        biv_model = BiventricularModel(MODEL_RESOURCE_DIR)

        # read control mesh
        biv_model.control_mesh = np.loadtxt(model_file, delimiter=',', skiprows=1, usecols=[0, 1, 2]).astype(float)

        meshes = {}

        for surface in Surface:
            if surface.name in surfaces_to_export:
                mesh_data = {surface.name: surface.value}
                add_valves(surface.name, mesh_data, closed_mesh) # add valves if want closed
                meshes[surface.name] = mesh_data

            # Special handling for RV endo
            if "RV_ENDOCARDIAL" in surfaces_to_export:
                rv_mesh = {"RV_SEPTUM": Surface.RV_SEPTUM.value, "RV_FREEWALL": Surface.RV_FREEWALL.value}
                add_valves("RV_ENDOCARDIAL", rv_mesh, closed_mesh)
                meshes["RV_ENDOCARDIAL"] = rv_mesh

        # Export all requested meshes
        for key, value in meshes.items():
            vertices, faces_mapped = get_verts_faces(biv_model, value)
            mesh_filename = f"{case_name}_{key}_{frame_name:03}{output_format}"
            export_mesh(output_format, output_directory, mesh_filename, vertices, faces_mapped, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script exports subdivded mesh surfaces from control meshes for a given set of cases.')
    parser.add_argument('-mdir', '--model_directory', type=Path, default=None,
                        help='Path to the folder of cases that each contain fitted control meshes')
    parser.add_argument('-o', '--output_directory', type=Path, 
                        help='Define the directory where you want to save new mesh surfaces to. Can be the same as input folder', default=None)
    parser.add_argument('-close', '--closed_mesh', action='store_true',
                        help='If set, will export closed meshes (with valves). If not set, will export open meshes (without valves)', default=False)
    parser.add_argument('-surfaces', '--surfaces_to_export', nargs='+', type=str, default=["LV_ENDOCARDIAL", "EPICARDIAL", "RV_ENDOCARDIAL"],
                        help='List of surfaces to export. Options: LV_ENDOCARDIAL, RV_SEPTUM, RV_FREEWALL, RV_EPICARDIAL, MITRAL_VALVE, AORTA_VALVE, TRICUSPID_VALVE, PULMONARY_VALVE. Default is LV, RV, and Epi.')
    
    parser.add_argument('-format', '--output_format', type=str, default='.vtk',
                        help='Output format for the exported meshes. Options: .vtk, .obj. Default is .vtk')
    args = parser.parse_args()

    if args.model_directory is not None:
        assert Path(args.model_directory).exists(), \
            f'model_directory does not exist. Cannot find {args.model_directory}!'
    else:
        logger.error('No model_directory provided. Exiting...')
        sys.exit(0)

    if args.output_directory is not None:
        output_folder = Path(args.output_directory)
        output_folder.mkdir(parents=True, exist_ok=True)

    else:
        logger.error('No output_directory provided. Exiting...')
        sys.exit(0)

    # set list of cases to process
    case_list = [c for c in os.listdir(args.model_directory) if os.path.isdir(os.path.join(args.model_directory, c))]
    case_dirs = [Path(args.model_directory, case).as_posix() for case in case_list if not case.startswith('.')]
    logger.info(f"Found {len(case_dirs)} cases to export.")

    # start processing...
    start_time = time.time()

    try:
        for dir in case_dirs:
            case = os.path.basename(dir)
            try:
                logger.info(f"Processing {case}")

                model_dir = Path(args.model_directory) / case

                output_folder_case = Path(output_folder) / case
                output_folder_case.mkdir(parents=True, exist_ok=True)

                export_from_control_mesh(model_dir, output_folder_case, args.closed_mesh, args.surfaces_to_export, args.output_format, case_name=case, logger=logger)

            except:
                logger.error(f"Could not process: {os.path.basename(case)}")

        logger.info(f"Total cases processed: {len(case_dirs)}")
        logger.info(f"Total time: {time.time() - start_time}")
        logger.success(f'Meshes are saved in {output_folder}')

    except KeyboardInterrupt:
        logger.info(f"Program interrupted by the user")
        sys.exit(0)