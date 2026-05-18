import os, sys
import numpy as np
import time
import plotly.graph_objs as go
from pathlib import Path
from plotly.offline import plot
import argparse
import nibabel as nib
import re
import fnmatch
import pyvista as pv
import cv2

from bivme.fitting.BiventricularModel import BiventricularModel
from bivme.fitting.GPDataSet import GPDataSet
from bivme.fitting.surface_enum import ContourType
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

from bivme import MODEL_RESOURCE_DIR

# This list of contours_to _plot was taken from Liandong Lee
contours_to_plot = [
    ContourType.LAX_RA,
    ContourType.LAX_LA,
    ContourType.SAX_RA,
    ContourType.SAX_LA,
    ContourType.SAX_LAA,
    ContourType.SAX_LPV,
    ContourType.SAX_RPV,
    ContourType.SAX_SVC,
    ContourType.SAX_IVC,
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
    ContourType.SAX_RV_OUTLET,
    ContourType.OUTLET_RV_FREEWALL,
    ContourType.OUTLET_RV_SEPTUM,
    ContourType.OUTLET_RV_EPICARDIAL,
    ContourType.AORTA_PHANTOM,
    ContourType.TRICUSPID_PHANTOM,
    ContourType.MITRAL_PHANTOM,
    ContourType.PULMONARY_PHANTOM,
    ContourType.EXCLUDED,
]

def _plot_one_frame(case, data_set, model, image_plot, num, my_logger):
    my_logger.info(f"Plotting frame {num:03d} for case {case}...")

    # Start with contours
    contour_plots = data_set.plot_dataset(contours_to_plot)
    data = contour_plots

    # Then add model surface if model is not None
    if model is not None:
        data += model.plot_surface(
                "rgb(0,127,0)", "rgb(0,127,127)", "rgb(127,0,0)", "all"
            )

    # Then add image plot if image_plot is not None
    if image_plot is not None:
        data += image_plot

    figure = go.Figure(data=data)
    figure.update_layout(
        paper_bgcolor='white',                
        title=f"Plot for {case} - Frame {num:03}",
    )
    figure.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)

    return figure

def prepare_gpdataset(gp_dir, filename_info, case, gp_suffix, si_suffix, num, my_logger):
    gp_rule = re.compile(fnmatch.translate(f"*GPFile_{gp_suffix}{num:03d}.txt"), re.IGNORECASE)
    filename = [Path(gp_dir) / name for name in os.listdir(gp_dir) if gp_rule.match(name)]

    if len(filename) == 0:
        my_logger.error(f"Cannot find GPfile for frame {num:03d}! Skipping this frame")
        return

    filename = filename[-1]

    data_set = GPDataSet(
        str(filename),
        str(filename_info),
        case,
        sampling=1,
        frame_num=num,
    )

    if not data_set.success:
        my_logger.error(f"Cannot initialize GPDataSet! Skipping this frame")
        return

    if data_set.slices == {}:
        my_logger.error(f"No slices found in GPDataSet! Check your SliceInfoFile is valid")
        return
    
    return data_set

def prepare_biv_model(model_path, num, my_logger):
    rule = re.compile(fnmatch.translate(f"*model_*frame*{num:03}.txt"), re.IGNORECASE)

    path_to_model = [Path(model_path) / name for name in os.listdir(model_path) if rule.match(name)]

    biventricular_model = BiventricularModel(MODEL_RESOURCE_DIR)
    control_points = np.loadtxt(path_to_model[-1], delimiter=',', skiprows=1, usecols=[0, 1, 2]).astype(np.float16)
    biventricular_model.update_control_mesh(control_points)

    return biventricular_model.copy()

def prepare_image_plots(image_path, data_set, image_grids, frame, shifts=None, output_path = None, logger = logger):
    image_plots = []

    for root, dirs, files in os.walk(image_path):
        # Only load images from the resized folder
        if 'resized-cropped-normalised' in root:
            continue

        for file in files:
            if file.endswith('.nii.gz'):
                nifti_file = os.path.join(root, file)
                
                # Load the NIfTI file
                img = nib.load(nifti_file)
                img_data = img.get_fdata()

                slice_number = int(file.split('_')[2])
                slice_type = file.split('_')[0]

                # Get shift for the current slice
                if shifts is None:
                    shift = [0,0]
                else:
                    idx = list(data_set.slices.keys()).index(slice_number)
                    shift = shifts[idx]

                # Extract the data for the current frame
                frame_data = img_data[:, :, frame]

                if slice_number not in image_grids.keys(): # Only compute the grid for this slice if it hasn't been computed before.
                    # Get slice
                    try:
                        current_slice = data_set.slices[slice_number]
                    except KeyError:
                        logger.warning(f"Slice number {slice_number} not found in dataset slices. Skipping this slice.")
                        continue

                    S = current_slice.position
                    X = current_slice.orientation[:3]
                    Y = current_slice.orientation[3:]
                    ps = current_slice.pixel_spacing

                    # construct affine transform
                    M = np.asarray([[X[0]*ps[0], Y[0]*ps[1], 0, S[0]],
                                [X[1]*ps[0], Y[1]*ps[1], 0, S[1]],
                                [X[2]*ps[0], Y[2]*ps[1], 0, S[2]],
                                [0, 0, 0, 1]])
                    
                    coords3d = np.zeros((frame_data.shape[0], frame_data.shape[1], 3))
                    pixel_vector = np.zeros((frame_data.shape[0]*frame_data.shape[1], 4))
                    for i in range(frame_data.shape[0]):
                        for j in range(frame_data.shape[1]): 
                            pixel_vector[i * frame_data.shape[1] + j] = np.array([j+shift[0]/ps[0], i+shift[1]/ps[1], 0, 1.0]) # Apply shift to x,y,z,1.0. Shifts contain scaling factor so must be divided by pixel spacing first

                    coords_vector = np.dot(M, pixel_vector.T).T
                    coords3d[:, :, 0] = coords_vector[:, 0].reshape((frame_data.shape[0], frame_data.shape[1]), order='C')
                    coords3d[:, :, 1] = coords_vector[:, 1].reshape((frame_data.shape[0], frame_data.shape[1]), order='C')
                    coords3d[:, :, 2] = coords_vector[:, 2].reshape((frame_data.shape[0], frame_data.shape[1]), order='C')

                    # Create meshgrid
                    gridX = np.reshape(coords3d[:, :, 0], (frame_data.shape[0], frame_data.shape[1]), order='C')
                    gridY = np.reshape(coords3d[:, :, 1], (frame_data.shape[0], frame_data.shape[1]), order='C')
                    gridZ = np.reshape(coords3d[:, :, 2], (frame_data.shape[0], frame_data.shape[1]), order='C')

                    gridX = gridX.astype(np.float16)
                    gridY = gridY.astype(np.float16)
                    gridZ = gridZ.astype(np.float16)

                    image_grids[slice_number] = (gridX, gridY, gridZ)
                
                else:
                    gridX = image_grids[slice_number][0]
                    gridY = image_grids[slice_number][1]
                    gridZ = image_grids[slice_number][2]


                # Apply CLAHE
                clahe = cv2.createCLAHE(tileGridSize=(1, 1))

                # Shift the pixel values to be all positive before applying CLAHE
                min_value = min(0, np.min(frame_data))
                frame_data += abs(min_value) 

                frame_data = clahe.apply(frame_data.astype(np.uint16))

                # Convert frame data to uint8 to save memory
                frame_data = (frame_data - np.min(frame_data)) / (np.max(frame_data) - np.min(frame_data)) * 255
                frame_data = frame_data.astype(np.uint8)

                image_plots.append(go.Surface(x=gridX, y=gridY, z=gridZ,
                    surfacecolor=frame_data, 
                    cmin=0, 
                    cmax=255,
                    showscale=False,
                    colorscale='gray',
                    name = f"Slice {slice_number} - {slice_type}",
                    showlegend=True,
                ))


                if output_path is not None:
                    # Export the image as a VTK file
                    gridX = gridX.astype(np.float32) # pyvista requires float32
                    gridY = gridY.astype(np.float32)
                    gridZ = gridZ.astype(np.float32)

                    grid = pv.StructuredGrid(gridX, gridY, gridZ)
                    grid.point_data.set_scalars(frame_data.flatten(order='F'))

                    # Create folder for the view if it doesn't exist
                    view_folder = os.path.join(output_path, slice_type)
                    os.makedirs(view_folder, exist_ok=True)

                    # Save the grid as a VTK file
                    save_name = f"{slice_type}_{slice_number}_{frame:03d}.vtk"
                    grid.save(os.path.join(view_folder, save_name))

    return image_plots, image_grids


def save_figure(figure, output_folder_html, case, num, my_logger):
    # Save the figure as an HTML file
    filepath = os.path.join(
        output_folder_html, f"{case}_frame_{num:03}.html"
    )

    plot(
        figure,
        filename=filepath,
        auto_open=False,
    )

    my_logger.success(f"{case} - frame {num:03} plot successfully saved to {filepath}")


def generate_html(case: str, gp_dir: str, out_dir: str ="./results/", gp_suffix: str ="", si_suffix: str ="", frames_to_fit: list[int]=[], my_logger: logger = logger, model_path = None, image_path = None, vtk_export_path = None, workers = 4) -> None:

    si_rule = re.compile(fnmatch.translate(f"*SliceInfoFile{si_suffix}*.txt"), re.IGNORECASE)
    filename_info = [Path(gp_dir) / Path(name) for name in os.listdir(Path(gp_dir)) if si_rule.match(name)]
    if len(filename_info) == 0:
        my_logger.error(f"Cannot find SliceInfoFile! Skipping this case")
        return -1
    filename_info = filename_info[-1]

    gp_rule = re.compile(fnmatch.translate(f"*GPFile_{gp_suffix}*.txt"), re.IGNORECASE)
    gp_files = [name for name in os.listdir(Path(gp_dir)) if gp_rule.match(name)]
    if len(gp_files) == 0:
        my_logger.error(f"Cannot find any GPFile with the specified suffix! Skipping this case")
        return -1
    
    if len(frames_to_fit) == 0:
        frames_to_fit = [int(f.split('_')[-1].replace('.txt','')) for f in gp_files] # fit all the frames available
    
    frames_to_fit = sorted(np.unique(frames_to_fit))

    # create a separate output folder for each patient
    output_folder = Path(out_dir) / case
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Create output folder for html files
    output_folder_html = Path(output_folder, f"html{gp_suffix}")
    output_folder_html.mkdir(exist_ok=True)

    start_time = time.time()
    my_logger.info(f"Preparing data for plotting")

    # Pre-prepare all the datasets to be plotted in parallel
    datasets = {}
    for num in frames_to_fit:
        data_set = prepare_gpdataset(gp_dir, filename_info, case, gp_suffix, si_suffix, num, my_logger)

        if data_set is not None:
            datasets[num] = data_set

    # Pre-prepare all the biventricular models to be plotted in parallel (if configured)
    if model_path is not None:
        models = {}
        for num in frames_to_fit:
            try:
                model = prepare_biv_model(model_path, num, my_logger) # Returns the biventricular model object for this frame, which contains the control mesh and can be used to plot the model surface later
            except Exception as e:
                my_logger.error(f"Error preparing biventricular model for frame {num:03}: {e}")
                model = None

            models[num] = model

    else:
        models = {num: None for num in frames_to_fit} # If no models to plot, create empty lists for each frame to simplify the plotting code later

    # Pre-prepare all the images to be plotted in parallel (if configured)
    if image_path is not None:
        image_grids = {} # keys are slice numbers, values are the corresponding grids for plotting the image slices. We store this to avoid recomputing the grids for each frame if they are the same.
        image_plots = {} # keys are frame numbers, values are the corresponding image plots for that frame 
        for num in frames_to_fit:
            if num in datasets.keys():
                data_set = datasets[num]
                try:
                    image_plot, image_grids = prepare_image_plots(image_path, data_set, image_grids, num, shifts=None, output_path = vtk_export_path, logger=my_logger) # Returns the image plots and the grids used for plotting (so that we don't have to recompute the grids for each frame if they are the same), and exports image VTK files to the specified path if configured
                except Exception as e:
                    my_logger.error(f"Error preparing image plots for frame {num:03}: {e}")
                    breakpoint()
                    image_plot = None

                image_plots[num] = image_plot
    else:
        image_plots = {num: None for num in frames_to_fit} # If no images to plot, create empty lists for each frame to simplify the plotting code later

    my_logger.info(f"[CHECKPOINT][PLOTTING] Preparing data for plotting took: {time.time()-start_time} seconds.")
    start_time = time.time()

    completed_plots = {}

    # Run the plotting in parallel for each frame
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_plot_one_frame, case, datasets[num], models[num], image_plots[num], num, my_logger): num for num in frames_to_fit}

        for fut in as_completed(futs):
            num = futs[fut]
            try:
                figure = fut.result()
                completed_plots[num] = figure

            except Exception as e:
                my_logger.error(f"Error plotting frame {num:03d}: {e}")
                completed_plots[num] = None

    my_logger.info(f"[CHECKPOINT][PLOTTING] Plotting frames took: {time.time()-start_time} seconds.")
    start_time = time.time()
    my_logger.info(f"Saving plots to HTML files...")

    # Save the completed plots
    for num, figure in completed_plots.items():
        if figure is not None:
            save_figure(figure, output_folder_html, case, num, my_logger)
            
    my_logger.info(f"[CHECKPOINT][PLOTTING] Saving HTML files took: {time.time()-start_time} seconds.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='This script generates an HTML plot of guidepoints and fitted models for given cases.')
    parser.add_argument('-o', '--output_folder', type=Path, default="./html",
                        help='Path to the output folder')
    parser.add_argument('-gp', '--gp_directory', type=Path, 
                        help='Define the directory containing guidepoint files', default=None)
    parser.add_argument('--gp_suffix', type =str, default = '', help='guidepoints to use if we do not want to fit all the models in the input folder')
    parser.add_argument('--si_suffix', type =str, default = '', help='Define slice info to use if multiple SliceInfo.txt file are available')
    parser.add_argument('-mdir', '--model_directory', type=Path,
                        help='Define the directory containing the model files', default = None)
    parser.add_argument('-idir', '--image_directory', type=Path,
                        help='Define the directory containing the image files', default = None)
    parser.add_argument('-vtkdir', '--vtk_directory', type=Path, default=None, help='Define the directory to which the image VTK files will be exported. If not provided, VTK files will not be exported.')

    args = parser.parse_args()

    # save config file to the output folder
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if args.model_directory is not None:
        assert Path(args.model_directory).exists(), \
            f'model_directory does not exist. Cannot find {args.model_directory}!'
        
    if args.gp_directory is not None:
        assert Path(args.gp_directory).exists(), \
            f'gp_directory does not exist. Cannot find {args.gp_directory}!'
    else:
        logger.error('No gp_directory provided. Exiting...')
        sys.exit(0)

    # set list of cases to process
    case_list = [c for c in os.listdir(args.gp_directory) if os.path.isdir(os.path.join(args.gp_directory, c))]
    case_dirs = [Path(args.gp_directory, case).as_posix() for case in case_list if not case.startswith('.')]
    logger.info(f"Found {len(case_dirs)} cases to plot.")

    # start processing...
    start_time = time.time()

    try:
        for dir in case_dirs:
            case = os.path.basename(dir)
            try:
                logger.info(f"Processing {case}")

                gp_dir = Path(args.gp_directory) / case

                if args.model_directory is not None:
                    model_dir = Path(args.model_directory) / case
                else:
                    model_dir = None

                if args.image_directory is not None:
                    image_dir = Path(args.image_directory) / case / "images"
                else:
                    image_dir = None

                if args.vtk_directory is not None:
                    vtk_export_dir = Path(args.vtk_directory) / case
                    vtk_export_dir.mkdir(parents=True, exist_ok=True)


                generate_html(case, gp_dir, out_dir=output_folder, gp_suffix=args.gp_suffix, si_suffix=args.si_suffix,
                                frames_to_fit=[], my_logger=logger, model_path = model_dir, image_path = image_dir, vtk_export_path=vtk_export_dir, workers=4)
            except:
                logger.error(f"Could not process: {os.path.basename(case)}")

        logger.info(f"Total cases processed: {len(case_dirs)}")
        logger.info(f"Total time: {time.time() - start_time}")
        logger.success(f'Plots are saved in {output_folder}')

    except KeyboardInterrupt:
        logger.info(f"Program interrupted by the user")
        sys.exit(0)