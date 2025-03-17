import os
import shutil
import pydicom
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import os
import pandas as pd
import argparse
import nibabel as nib
import shutil
from pathlib import Path
import multiprocessing as mp
from functools import partial
from datetime import datetime
import logging
from time import sleep, time

class ViewSelector:

    def __init__(self, src : Path, model_path : Path, patient_name : str, output_dir : Path):
        self.src = src
        self.model = joblib.load(model_path)
        self.patient_name = patient_name
        self.df = None
        self.unique_df = {}
        self.output_dir = output_dir
        self.sorted_dict = {}

    def get_dicom_header(self, dicom_loc : Path):

        # read dicom file and return header information and image
        ds = pydicom.read_file(dicom_loc, force=True)
        # get patient, study, and series information
        patient_id = ds.get("PatientID", "NA")
        modality = ds.get("Modality","NA")
        instance_number = ds.get("InstanceNumber","NA")
        series_instance_uid = ds.get("SeriesInstanceUID","NA")
        series_number = ds.get('SeriesNumber', 'NA')
        image_position_patient = ds.get("ImagePositionPatient", 'NA')
        image_orientation_patient = ds.get("ImageOrientationPatient", 'NA')
        pixel_spacing = ds.get("PixelSpacing", 'NA')
        echo_time = ds.get("EchoTime", 'NA')
        repetition_time = ds.get("RepetitionTime", 'NA')
        trigger_time = float(ds.get('TriggerTime', 'NA'))
        image_dimension = [ds.get('Rows', 'NA'), ds.get('Columns', 'NA')]
        slice_thickness = ds.get('SliceThickness', 'NA')
        slice_location = ds.get('SliceLocation', 'NA')

        # store image data
        array = ds.pixel_array

        return patient_id, dicom_loc, modality, instance_number, series_instance_uid, series_number , tuple(image_position_patient), image_orientation_patient, pixel_spacing, echo_time, repetition_time, trigger_time, image_dimension, slice_thickness, array, slice_location

    def predict_views(self, export_to_csv = False):


        directory = self.src

        p = directory.glob('**/*')

        unsorted_list = [str(file) for file in p if ".dcm" in str(file)]

        #unsorted_list = [os.path.join(root, file) for root, _, files in os.walk(self.src) for file in files if ".dcm" in file ]
        output = []
        for dicom_loc in unsorted_list:
            try:
                output.append(self.get_dicom_header(dicom_loc))
            except:
                continue

        # generated pandas dataframe to store information from headers
        self.df = pd.DataFrame(output, columns=['Patient ID',
                                                'Filename',
                                                'Modality',
                                                'Instance Number',                                   
                                                'Series InstanceUID',
                                                'Series Number',
                                                'Image Position Patient',
                                                'Image Orientation Patient',
                                                'Pixel Spacing', 
                                                'Echo Time',
                                                'Repetition Time',
                                                'Trigger Time',
                                                'Image Dimension',
                                                'Slice Thickness',
                                                'image',
                                                'Slice Location'])

        self.sort_dicom_per_series()
        self.predict(export_to_csv)

        if len(self.unique_df) == 0:
            logging.warning(f'DataFrame is empty for {self.patient_name}')

        if export_to_csv and len(self.unique_df) > 0:
            output_path = self.output_dir / Path(self.patient_name)#os.path.join(self.output_dir, self.patient_name)
            self.unique_df.to_csv(f'{output_path}/view-prediction.csv', sep='\t')

    def predict(self, export_to_csv):
        # Each series is a separate row in the dataframe
        # Merge frames for each series
                #self.merge_dicom_frames_and_predict(export_to_csv)
        
        directory = self.output_dir / self.patient_name
        p = directory.glob('**/*')
        files = [x for x in p if x.is_file()]

        avg_participant = [0.0, 0.0, 0.0]
        number_of_average = 0
        for file in files:
            try:
                ds = pydicom.dcmread(file)
                p2 = [ds.Rows /2, ds.Columns /2]
                pixel_spacing = [ds.PixelSpacing[0], ds.PixelSpacing[1]]
                image_position = [ds.ImagePositionPatient[i] for i in range(3) ]
                image_orientation = [ds.ImageOrientationPatient[i] for i in range(6) ]
                x, y, z = from_2d_to_3d(p2, image_orientation, image_position, pixel_spacing)
                avg_participant[0] += x
                avg_participant[1] += y
                avg_participant[2] += z
                number_of_average += 1
            except:
                continue

        if number_of_average == 0:
            return

        avg_participant[0] = avg_participant[0] / number_of_average
        avg_participant[1] = avg_participant[1] / number_of_average
        avg_participant[2] = avg_participant[2] / number_of_average

        sids = [n for n in os.listdir(self.output_dir / self.patient_name / 'temp')]

        output_dataframe = []
        count = 0
        for ids in sids:
            

            dcm = [n for n in os.listdir(self.output_dir / self.patient_name / 'temp' / ids) if 'dcm' in n]

            ds = pydicom.dcmread(self.output_dir / self.patient_name / 'temp' / ids / dcm[0])

            p2 = [ds.Rows /2, ds.Columns /2]
            pixel_spacing = [ds.PixelSpacing[0], ds.PixelSpacing[1]]
            image_position = [ds.ImagePositionPatient[i] for i in range(3) ]
            image_orientation = [ds.ImageOrientationPatient[i] for i in range(6) ]
            x, y, z = from_2d_to_3d(p2, image_orientation, image_position, pixel_spacing)

            my_vector = np.array([avg_participant[0] - x, 
                                  avg_participant[1] - y,
                                  avg_participant[2] - z])
            magnitude = np.linalg.norm(my_vector)
            normalized_vector = my_vector / magnitude

            predictors = np.array([float(ds.EchoTime),  
                                        float(ds.ImageOrientationPatient[0]), 
                                        float(ds.ImageOrientationPatient[1]), 
                                        float(ds.ImageOrientationPatient[2]), 
                                        float(ds.ImageOrientationPatient[3]), 
                                        float(ds.ImageOrientationPatient[4]), 
                                        float(ds.ImageOrientationPatient[5]), 
                                        float(normalized_vector[0]), 
                                        float(normalized_vector[1]), 
                                        float(normalized_vector[2]),   
                                        float(ds.ImagePositionPatient[0]), 
                                        float(ds.ImagePositionPatient[1]), 
                                        float(ds.ImagePositionPatient[2]),           
                                        float(ds.RepetitionTime),
                                        float(ds.SliceThickness)])
            
            scaler = self.model.scaler
            scaled_predictors = scaler.transform(predictors.reshape(1, -1))

            y_pred = self.model.predict(scaled_predictors)

            temp_dir = os.path.join(self.output_dir, self.patient_name, f'{ds.SeriesNumber}_{y_pred[0]}_frame_Volume_Sequence_by_InstanceNumber-{count}')          

            key = f'{ids}'
            write_nifti(self.sorted_dict[key], os.path.join(self.output_dir, self.patient_name, f"{Path(temp_dir).name}.nii.gz"))
            count = count +1

            if export_to_csv:
                output_dataframe.append([Path(ids).name, 
                                            ds.InstanceNumber,   
                                            ds.SeriesInstanceUID, 
                                            ds.SeriesNumber, 
                                            len(self.sorted_dict[key]),
                                            y_pred[0]])


        if export_to_csv:
            self.unique_df = pd.DataFrame(output_dataframe, columns=['Patient ID',
                                                                    'Instance Number',
                                                                    'Series InstanceUID',
                                                                    'Series Number',
                                                                    'Number of Frames',
                                                                    'Predicted view'])

        # delete temp folder
        shutil.rmtree(os.path.join(self.output_dir / self.patient_name / 'temp'))

    def sort_dicom_per_series(self):
        # Each series is a separate row in the dataframe
        # Merge frames for each series
        unique_series = self.df[['Series Number', 'Image Position Patient']].drop_duplicates()

        logging.info(f"{len(unique_series)} unique series found for {self.patient_name}")
        count = 0

        (self.output_dir / Path(self.patient_name)).mkdir(parents=True, exist_ok=True)
        (self.output_dir / Path(self.patient_name) / 'temp').mkdir(parents=True, exist_ok=True)

        for _, row in unique_series.iterrows():

            series_rows = self.df.loc[(self.df['Series Number'] == row['Series Number']) & (self.df['Image Position Patient'] == row['Image Position Patient'])]

            if len(series_rows) < 10: # unlikely to be a cine
                logging.warning(f"Removing series {row['Series Number']} for patient {self.patient_name} - less than 10 frames")
                continue

            series_rows = series_rows.sort_values('Trigger Time')
            pos = series_rows['Image Position Patient'].values[0]

            key = f'{series_rows["Series Number"].values[0]}_{pos[2]}'
            dcm_path = os.path.join(self.output_dir, self.patient_name,'temp',key)
            os.makedirs(dcm_path, exist_ok=True) 

            count = 0
            for name in series_rows['Filename']:
                num = int(series_rows['Trigger Time'].values[count])
                shutil.copy(name, dcm_path / Path(f'{num:05}.dcm')) 
                count += 1

            self.sorted_dict[key] = series_rows
            
def from_2d_to_3d(
    p2, image_orientation, image_position, pixel_spacing
):
    """# Convert indices of a pixel in a 2D image in space to 3D coordinates.
    #	Inputs
    #		image_orientation
    #		image_position
    #		pixel_spacing
    #		subpixel_resolution
    #	Outputs
    #		P3:  3D points
    """
    # if points2D.
    points2D = np.array(p2)

    S = np.eye(4)
    S[0, 0] = pixel_spacing[1]
    S[1, 1] = pixel_spacing[0]
    S = np.matrix(S)

    R = np.identity(4)
    R[0:3, 0] = image_orientation[
        0:3
    ]  # col direction, i.e. increases with row index i
    R[0:3, 1] = image_orientation[
        3:7
    ]  # row direction, i.e. increases with col index j
    R[0:3, 2] = np.cross(R[0:3, 0], R[0:3, 1])

    T = np.identity(4)
    T[0:3, 3] = image_position

    F = np.identity(4)
    F[0:1, 3] = -0.5

    T = np.dot(T, R)
    T = np.dot(T, S)
    Transformation = np.dot(T, F)

    pts = np.ones((len(points2D), 4))
    pts[:, 0:2] = points2D
    pts[:, 2] = [0] * len(points2D)
    pts[:, 3] = [1] * len(points2D)

    Px = np.dot(Transformation, pts.T)
    p3 = Px[0:3, :] / (np.vstack((Px[3, :], np.vstack((Px[3, :], Px[3, :])))))
    p3 = p3.T

    return p3[0, 0], p3[0, 1], p3[0, 2]

def write_nifti(df, output_file):

    ref = df.iloc[0, :]
    image_orientation = [float(ior) for ior in ref['Image Orientation Patient']]
    image_position = [float(ior) for ior in ref['Image Position Patient']]
    pixel_spacing = [float(ior) for ior in ref['Pixel Spacing']]
    img_dim = [int(ior) for ior in ref['Image Dimension']]
    slice_thickness = float(ref['Slice Thickness'])

    F11, F21, F31 = image_orientation[3:]
    F12, F22, F32 = image_orientation[:3]

    step = -np.cross(image_orientation[3:], image_orientation[:3]) * slice_thickness

    delta_r, delta_c = pixel_spacing
    Sx, Sy, Sz = image_position

    #affine = np.array(
    #    [
    #        [-F12 * delta_c, -F11 * delta_r, -step[0], -Sx],
    #        [-F22 * delta_c, -F21 * delta_r, -step[1], -Sy],
    #        [F32 * delta_c ,  F31 * delta_r,  step[2],  Sz],
    #        [0, 0, 0, 1]
    #    ]
    #)

    affine = np.array(
        [
            [-F11 * delta_r, -F12 * delta_c, -step[0], -Sx],
            [-F21 * delta_r, -F22 * delta_c, -step[1], -Sy],
            [F31 * delta_r, F32 * delta_c, step[2], Sz],
            [0, 0, 0, 1]
        ]
    )

    new_nifti = np.zeros((img_dim[0], img_dim[1], df.shape[0]))
    count = 0
    for _, row in df.iterrows():
        new_nifti[:,:, count] = row.image
        count +=1

    img_nii = nib.Nifti1Image(new_nifti.astype(np.float32), affine)
    nib.save(img_nii, output_file)

def select_views(patient_name: str, model: Path, output_dir: Path, root: Path, export_to_csv: bool = False):

    case_src = root / patient_name #/ patient_name
    viewSelector = ViewSelector(case_src, model, patient_name, output_dir)
    viewSelector.predict_views(export_to_csv)

if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Auto-segmentation pipeline for SCMR data')
    parser.add_argument('-b', '--base-folder', type=Path, required=True,
                        help='Base directory containing patient folders')
    parser.add_argument('-o', '--output_folder', type=Path, default='./output',
                        help='Output folder')
    parser.add_argument('-ckpt', '--path_to_model', type=Path, default='./metadata-based_model.joblib',
                        help='Path to checkpoints')
    parser.add_argument("-num_workers", type=int, default=4,
                    help="Number of workers, [default: 4]")
    parser.add_argument('-csv',"--export_to_csv", action="store_true",
                    help="Export csv with results from view detection")   
    args = parser.parse_args()

    model = args.path_to_model
    num_workers = args.num_workers

    assert Path(args.base_folder).exists(), \
        f'base-folder does not exist. Cannot find {args.base_folder}!'

    log_dir = os.path.join(args.output_folder,'log')

    time_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(log_dir, exist_ok=True)

    log_txt_file = os.path.join(log_dir, "view_detection_" + time_file + ".txt")


    # Set up logging for the main process
    logging.basicConfig(filename=log_txt_file , level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    
    # Set up multiprocessing logging
    mp.log_to_stderr(logging.INFO)

    logging.info("Starting view detection\n")

    caselist = [name for name in os.listdir(args.base_folder)]
  
    logging.info(f"DEBUG: patient list: {caselist}")

    preprocess_tr = partial(select_views, model=model, output_dir=args.output_folder, root=args.base_folder, export_to_csv =args.export_to_csv)

    t1a = time()

    with mp.Pool(num_workers) as p:
        with tqdm(total=len(caselist)) as pbar:
            pbar.set_description("Detecting views")
            [pbar.update() for i, _ in tqdm(enumerate(p.imap_unordered(preprocess_tr, caselist)))]

    t1b = time()
    logging.info(f"DEBUG studies: {len(caselist)}")
    logging.info(f"DEBUG Time taken: %.3f sec" % (t1b - t1a))
    logging.info("Closing view_detection_log_{}.txt".format(time_file))