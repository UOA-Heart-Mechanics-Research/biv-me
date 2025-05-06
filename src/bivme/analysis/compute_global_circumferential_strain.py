# Author: Anna Mira
# Reviewed by Laura Dal Toso on 18/08/2022
# Reviewed by Charlene Mauger on 22/08/2024

# This script computes the global circumferential strain from the models output
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import time
import csv
from loguru import logger
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import argparse
import fnmatch
from rich.progress import Progress
from bivme.meshing.mesh import Mesh
from bivme import MODEL_RESOURCE_DIR
import scipy.io

def calculate_circumferential_strain(case_name: str, model_file: os.PathLike, biv_model_folder: os.PathLike, precision: int) -> dict:
    """
    # Author: ldt
    # Date: 18/08/22

    This functions measures various strain metrics, from the models fitted at ES and ED.
    Input:
        - folder: folder where the Model.txt files are saved
        - output_file: csv file where the srtain measures should be saved

    """

    # read GP file
    control_points = np.loadtxt(model_file, delimiter=',', skiprows=1, usecols=[0, 1, 2]).astype(float)

    frame_name = re.search(r'Frame_(\d+)\.txt', str(model_file), re.IGNORECASE)[1]
    # assign values to dict
    results_dict = {'case': case_name, 'frame': frame_name} | {
        k: np.nan for k in ['lv_gcs_apex', 'lv_gcs_mid', 'lv_gcs_base', 'rvfw_gcs_apex', 'rvfw_gcs_mid', 'rvfw_gcs_base', 'rvs_gcs_apex', 'rvs_gcs_mid', 'rvs_gcs_base']
    }

    subdivision_matrix_file = biv_model_folder / "subdivision_matrix_sparse.mat"
    assert subdivision_matrix_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {subdivision_matrix_file} file!"

    circumferential_points_file = biv_model_folder / 'cs_points.txt'
    assert circumferential_points_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {circumferential_points_file} file!"

    cs_points = pd.read_table(circumferential_points_file, sep='\t')

    if control_points.shape[0] > 0:

        subdivision_matrix = scipy.io.loadmat(subdivision_matrix_file)['S'].toarray()

        vertices = np.dot(subdivision_matrix, control_points)

        lv_gcs_apex = (cs_points[(cs_points.View == "APEX") & (cs_points.Surface == "LV")].Index).to_numpy()
        lv_gcs_apex_vertices = vertices[lv_gcs_apex, :]
        lv_gcs_apex = np.linalg.norm(lv_gcs_apex_vertices[1:, ]-lv_gcs_apex_vertices[:-1, ], axis=1)
        results_dict['lv_gcs_apex'] = round(np.sum(lv_gcs_apex), precision)

        lv_gcs_mid_idx = (cs_points[(cs_points.View == "MID") & (cs_points.Surface == "LV")].Index).to_numpy()
        lv_gcs_mid_vertices = vertices[lv_gcs_mid_idx, :]
        lv_gcs_mid = np.linalg.norm(lv_gcs_mid_vertices[1:, :] - lv_gcs_mid_vertices[:-1, :], axis=1)
        results_dict['lv_gcs_mid'] = round(np.sum(lv_gcs_mid), precision)

        lv_gcs_base_idx = (cs_points[(cs_points.View == "BASE") & (cs_points.Surface == "LV")].Index).to_numpy()
        lv_gcs_base_vertices = vertices[lv_gcs_base_idx, :]
        lv_gcs_base = np.linalg.norm(lv_gcs_base_vertices[1:, :] - lv_gcs_base_vertices[:-1, :], axis=1)
        results_dict['lv_gcs_base'] = round(np.sum(lv_gcs_base), precision)

        rvfw_gcs_apex = (cs_points[(cs_points.View == "APEX") & (cs_points.Surface == "RVFW")].Index).to_numpy()
        rvfw_gcs_apex_vertices = vertices[rvfw_gcs_apex, :]
        rvfw_gcs_apex = np.linalg.norm(rvfw_gcs_apex_vertices[1:, ]-rvfw_gcs_apex_vertices[:-1, ], axis=1)
        results_dict['rvfw_gcs_apex'] = round(np.sum(rvfw_gcs_apex), precision)

        rvfw_gcs_mid_idx = (cs_points[(cs_points.View == "MID") & (cs_points.Surface == "RVFW")].Index).to_numpy()
        rvfw_gcs_mid_vertices = vertices[rvfw_gcs_mid_idx, :]
        rvfw_gcs_mid = np.linalg.norm(rvfw_gcs_mid_vertices[1:, :] - rvfw_gcs_mid_vertices[:-1, :], axis=1)
        results_dict['rvfw_gcs_mid'] = round(np.sum(rvfw_gcs_mid), precision)

        rvfw_gcs_base_idx = (cs_points[(cs_points.View == "BASE") & (cs_points.Surface == "RVFW")].Index).to_numpy()
        rvfw_gcs_base_vertices = vertices[rvfw_gcs_base_idx, :]
        rvfw_gcs_base = np.linalg.norm(rvfw_gcs_base_vertices[1:, :] - rvfw_gcs_base_vertices[:-1, :], axis=1)
        results_dict['rvfw_gcs_base'] = round(np.sum(rvfw_gcs_base), precision)

        rvs_gcs_apex = (cs_points[(cs_points.View == "APEX") & (cs_points.Surface == "RVS")].Index).to_numpy()
        rvs_gcs_apex_vertices = vertices[rvs_gcs_apex, :]
        rvs_gcs_apex = np.linalg.norm(rvs_gcs_apex_vertices[1:, ]-rvs_gcs_apex_vertices[:-1, ], axis=1)
        results_dict['rvs_gcs_apex'] = round(np.sum(rvs_gcs_apex), precision)

        rvs_gcs_mid_idx = (cs_points[(cs_points.View == "MID") & (cs_points.Surface == "RVS")].Index).to_numpy()
        rvs_gcs_mid_vertices = vertices[rvs_gcs_mid_idx, :]
        rvs_gcs_mid = np.linalg.norm(rvs_gcs_mid_vertices[1:, :] - rvs_gcs_mid_vertices[:-1, :], axis=1)
        results_dict['rvs_gcs_mid'] = round(np.sum(rvs_gcs_mid), precision)

        rvs_gcs_base_idx = (cs_points[(cs_points.View == "BASE") & (cs_points.Surface == "RVS")].Index).to_numpy()
        rvs_gcs_base_vertices = vertices[rvs_gcs_base_idx, :]
        rvs_gcs_base = np.linalg.norm(rvs_gcs_base_vertices[1:, :] - rvs_gcs_base_vertices[:-1, :], axis=1)
        results_dict['rvs_gcs_base'] = round(np.sum(rvs_gcs_base), precision)

    else:
        logger.error(f"No strain calculated for {model_file} please check the model file")

    return results_dict


if __name__ == "__main__":
    biv_resource_folder = MODEL_RESOURCE_DIR

    # parse command-line argument
    parser = argparse.ArgumentParser(description="Global circumferential strain calculation")
    parser.add_argument('-mdir', '--model_dir', type=Path, help='path to biv models')
    parser.add_argument('-o', '--output_path', type=Path, help='output path', default="./")
    parser.add_argument("-b", '--biv_model_folder', default=biv_resource_folder,
                        help="folder containing subdivision matrices"
                             f" (default: {biv_resource_folder})")
    parser.add_argument("-pat", '--patterns', default="*",
                        help="folder patterns to include (default '*')")
    parser.add_argument("-ed", '--ed_frame', default=0, type=int,
                        help="ED frame")
    parser.add_argument("-p", '--precision', type=int, default=2,
                        help="Output precision")
    args = parser.parse_args()

    fieldnames = ['name', 'frame', 'lv_gcs_apex', 'lv_gcs_mid', 'lv_gcs_base', 'rvfw_gcs_apex', 'rvfw_gcs_mid', 'rvfw_gcs_base', 'rvs_gcs_apex', 'rvs_gcs_mid', 'rvs_gcs_base']

    assert args.model_dir.exists(), \
        f"model_dir does not exist."

    if not args.output_path.exists():
        args.output_path.mkdir(parents=True, exist_ok=True) 

    folders = [p.name for p in Path(args.model_dir).glob(args.patterns) if os.path.isdir(p)]
    logger.info(f"Found {len(folders)} model folders.")

    output_cs_strain_file = args.output_path / 'global_circumferential_strain.csv'
    with open(output_cs_strain_file, 'w', newline='') as f:
        # create output file and write header
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        logger.info(f"Created {args.output_path} for the results.")

    for i, folder in enumerate(folders):
        rule = re.compile(fnmatch.translate("*model_frame*.txt"), re.IGNORECASE)
        models = [args.model_dir / folder / Path(name) for name in os.listdir(args.model_dir / folder) if
                  rule.match(name)]

        models = sorted(models)
        logger.info(f"Processing {str(args.model_dir / folder)} ({i + 1}/{len(folders)})")
        with Progress(transient=True) as progress:
            task = progress.add_task(f"Calculating strains", total=len(models))
            console = progress

            strain_values = [calculate_circumferential_strain(folder, biv_model_file, biv_resource_folder, args.precision) for biv_model_file in models]
            strain_values = pd.DataFrame(strain_values)

            with open(output_cs_strain_file, 'a', newline='') as file:
                # print out measurements in spreadsheet
                strain_writer = csv.writer(file)
                for idx, biv_model_file in enumerate(models):

                    strain_writer.writerow([folder,
                                            strain_values['frame'].iloc[idx],
                                            (strain_values['lv_gcs_apex'].iloc[idx] -
                                                   strain_values['lv_gcs_apex'].iloc[args.ed_frame]) /
                                            strain_values['lv_gcs_apex'].iloc[args.ed_frame],
                                            (strain_values['lv_gcs_mid'].iloc[idx] -
                                                   strain_values['lv_gcs_mid'].iloc[args.ed_frame]) /
                                            strain_values['lv_gcs_mid'].iloc[args.ed_frame],
                                            (strain_values['lv_gcs_base'].iloc[idx] -
                                                   strain_values['lv_gcs_base'].iloc[args.ed_frame]) /
                                            strain_values['lv_gcs_base'].iloc[args.ed_frame],
                                            (strain_values['rvfw_gcs_apex'].iloc[idx] -
                                                   strain_values['rvfw_gcs_apex'].iloc[args.ed_frame]) /
                                            strain_values['rvfw_gcs_apex'].iloc[args.ed_frame],
                                            (strain_values['rvfw_gcs_mid'].iloc[idx] -
                                                   strain_values['rvfw_gcs_mid'].iloc[args.ed_frame]) /
                                            strain_values['rvfw_gcs_mid'].iloc[args.ed_frame],
                                            (strain_values['rvfw_gcs_base'].iloc[idx] -
                                                   strain_values['rvfw_gcs_base'].iloc[args.ed_frame]) /
                                            strain_values['rvfw_gcs_base'].iloc[args.ed_frame],
                                            (strain_values['rvs_gcs_apex'].iloc[idx] -
                                                   strain_values['rvs_gcs_apex'].iloc[args.ed_frame]) /
                                            strain_values['rvs_gcs_apex'].iloc[args.ed_frame],
                                            (strain_values['rvs_gcs_mid'].iloc[idx] -
                                                   strain_values['rvs_gcs_mid'].iloc[args.ed_frame]) /
                                            strain_values['rvs_gcs_mid'].iloc[args.ed_frame],
                                            (strain_values['rvs_gcs_base'].iloc[idx] -
                                                   strain_values['rvs_gcs_base'].iloc[args.ed_frame]) /
                                            strain_values['rvs_gcs_base'].iloc[args.ed_frame]
                                            ])
            progress.advance(task)

    logger.success(f"Done. Results are saved in {output_cs_strain_file}")