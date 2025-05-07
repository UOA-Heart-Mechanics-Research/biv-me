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

def calculate_longitudinal_strain(case_name: str, model_file: os.PathLike, biv_model_folder: os.PathLike, precision: int) -> dict:
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
        k: np.nan for k in ['lv_gls_2ch', 'lv_gls_4ch', 'rvs_gls_4ch', 'rvfw_gls_4ch']
    }

    subdivision_matrix_file = biv_model_folder / "subdivision_matrix_sparse.mat"
    assert subdivision_matrix_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {subdivision_matrix_file} file!"

    longitudinal_points_file = biv_model_folder / 'ls_points.txt'
    assert longitudinal_points_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {longitudinal_points_file} file!"

    ls_points = pd.read_table(longitudinal_points_file, sep='\t')

    if control_points.shape[0] > 0:

        subdivision_matrix = scipy.io.loadmat(subdivision_matrix_file)['S'].toarray()

        vertices = np.dot(subdivision_matrix, control_points)

        lv_gls_2ch_idx = (ls_points[(ls_points.View == "2CH") & (ls_points.Surface == "LV")].Index).to_numpy()
        lv_gls_2ch_vertices = vertices[lv_gls_2ch_idx, :]
        lv_gls_2ch = np.linalg.norm(lv_gls_2ch_vertices[1:, ]-lv_gls_2ch_vertices[:-1, ], axis=1)
        results_dict['lv_gls_2ch'] = round(np.sum(lv_gls_2ch), precision)

        lv_gls_4ch_idx = (ls_points[(ls_points.View == "4CH") & (ls_points.Surface == "LV")].Index).to_numpy()
        lv_gls_4ch_vertices = vertices[lv_gls_4ch_idx, :]
        lv_gls_4ch = np.linalg.norm(lv_gls_4ch_vertices[1:, :] - lv_gls_4ch_vertices[:-1, :], axis=1)
        results_dict['lv_gls_4ch'] = round(np.sum(lv_gls_4ch), precision)

        rvs_gls_4ch_idx = (ls_points[(ls_points.View == "4CH") & (ls_points.Surface == "RVS")].Index).to_numpy()
        rvs_gls_4ch_vertices = vertices[rvs_gls_4ch_idx, :]
        rvs_gls_4ch = np.linalg.norm(rvs_gls_4ch_vertices[1:, :] - rvs_gls_4ch_vertices[:-1, :], axis=1)
        results_dict['rvs_gls_4ch'] = round(np.sum(rvs_gls_4ch), precision)

        rvfw_gls_4ch_idx = (ls_points[(ls_points.View == "4CH") & (ls_points.Surface == "RVFW")].Index).to_numpy()
        rvfw_gls_4ch_vertices = vertices[rvfw_gls_4ch_idx, :]
        rvfw_gls_4ch = np.linalg.norm(rvfw_gls_4ch_vertices[1:, :] - rvfw_gls_4ch_vertices[:-1, :], axis=1)
        results_dict['rvfw_gls_4ch'] = round(np.sum(rvfw_gls_4ch), precision)

    else:
        logger.error(f"No strain calculated for {model_file} please check the model file")

    return results_dict


if __name__ == "__main__":

    biv_resource_folder = MODEL_RESOURCE_DIR

    # parse command-line argument
    parser = argparse.ArgumentParser(description="Global longitudinal strain calculation")
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

    fieldnames = ['name', 'frame', 'lv_gls_2ch', 'lv_gls_4ch', 'rvs_gls_4ch', 'rvfw_gls_4ch']

    assert args.model_dir.exists(), \
        f"model_dir does not exist."

    if not args.output_path.exists():
        args.output_path.mkdir(parents=True, exist_ok=True) 

    folders = [p.name for p in Path(args.model_dir).glob(args.patterns) if os.path.isdir(p)]
    logger.info(f"Found {len(folders)} model folders.")

    output_ls_strain_file = args.output_path / 'global_longitudinal_strain.csv'
    with open(output_ls_strain_file, 'w', newline='') as f:
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

            strain_values = [calculate_longitudinal_strain(folder, biv_model_file, biv_resource_folder, args.precision) for biv_model_file in models]
            strain_values = pd.DataFrame(strain_values)

            with open(output_ls_strain_file, 'a', newline='') as file:
                # print out measurements in spreadsheet
                strain_writer = csv.writer(file)
                for idx, biv_model_file in enumerate(models):

                    strain_writer.writerow([folder,
                                            strain_values['frame'].iloc[idx],
                                            (strain_values['lv_gls_2ch'].iloc[idx] -
                                                   strain_values['lv_gls_2ch'].iloc[args.ed_frame]) /
                                            strain_values['lv_gls_2ch'].iloc[args.ed_frame],
                                            (strain_values['lv_gls_4ch'].iloc[idx] -
                                                   strain_values['lv_gls_4ch'].iloc[args.ed_frame]) /
                                            strain_values['lv_gls_4ch'].iloc[args.ed_frame],
                                            (strain_values['rvs_gls_4ch'].iloc[idx] -
                                                   strain_values['rvs_gls_4ch'].iloc[args.ed_frame]) /
                                            strain_values['rvs_gls_4ch'].iloc[args.ed_frame],
                                            (strain_values['rvfw_gls_4ch'].iloc[idx] -
                                                   strain_values['rvfw_gls_4ch'].iloc[args.ed_frame]) /
                                            strain_values['rvfw_gls_4ch'].iloc[args.ed_frame]
                                            ])
            progress.advance(task)

    logger.success(f"Done. Results are saved in {output_ls_strain_file}")