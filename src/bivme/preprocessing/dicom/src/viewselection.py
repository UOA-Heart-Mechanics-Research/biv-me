import os
import shutil
import pydicom
import numpy as np
import pandas as pd
import PIL.Image as Image
import joblib
import os
import pandas as pd
import shutil
from pathlib import Path

from bivme.preprocessing.dicom.src.utils import clean_text, from_2d_to_3d

class ViewSelector:

    def __init__(self, src, dst, model, csv_path, my_logger):
        self.src = src
        self.dst = dst
        self.model = model
        self.csv_path = csv_path
        self.df = None
        self.my_logger = my_logger

    def load_predictions(self):
        self.store_dicom_info()
        self.prepare_data_for_prediction()
        self.write_sorted_pngs()

    def prepare_data_for_prediction(self):
        self.store_dicom_info()

        # Destination for view classification
        dir_view_classification = os.path.join(self.dst, 'view-classification')
        os.makedirs(dir_view_classification, exist_ok=True)

        # Write images to png
        dir_unsorted = os.path.join(dir_view_classification, 'unsorted')
        
        if not os.path.exists(dir_unsorted):
            os.makedirs(dir_unsorted)
        else: 
            # Clear directory
            for file in os.listdir(dir_unsorted):
                os.remove(os.path.join(dir_unsorted, file))

        # Generate dummy annotations file
        annotations = []

        for i, row in self.df.iterrows():
            img = row['Img']
            for frame in range(img.shape[0]):
                img_data = img[frame, :, :]

                # Transpose
                img_data = np.transpose(img_data)
                
                # Cast to uint8
                # Remap to 0-255
                img_data = img_data - np.min(img_data)
                img_data = img_data / np.max(img_data) * 255
                img_data = img_data.astype(np.uint8)

                img_data = np.stack((img_data,)*3, axis=-1)

                # Save as png
                img_data = Image.fromarray(img_data)
                save_path = os.path.join(dir_unsorted, f'{row["Series Number"]}_{frame}.png')
                img_data.save(save_path)
        
                annotations.append([os.path.basename(save_path), 0])
        
        # Write dummy annotations to file
        test_annotations = os.path.join(dir_view_classification, 'test_annotations.csv')
        test_annotations_df = pd.DataFrame(annotations, columns=['image_name', 'view'])
        test_annotations_df.to_csv(test_annotations, index=False)

        self.my_logger.info(f"Data prepared for view prediction. {len(self.df)} image series found.")

    def write_sorted_pngs(self):
        # Load view predictions
        view_predictions = pd.read_csv(self.csv_path)

        # Unsorted directory
        dir_unsorted = os.path.join(self.dst, 'view-classification', 'unsorted')

        # Destination for view classification
        dir_sorted = os.path.join(self.dst, 'view-classification', 'sorted')
        if not os.path.exists(dir_sorted):
            os.makedirs(dir_sorted)
        else:
            # Clear directory
            shutil.rmtree(dir_sorted)
            os.makedirs(dir_sorted)
        
        # Move images to respective folders
        for i, row in view_predictions.iterrows():
            series_number = row['Series Number']
            view = row['Predicted View']

            os.makedirs(os.path.join(dir_sorted, view), exist_ok=True)
            
            # Grab images
            series_images = [f for f in os.listdir(dir_unsorted) if f.startswith(f'{series_number}_')]
            for img in series_images:
                src = os.path.join(dir_unsorted, img)
                dst = os.path.join(dir_sorted, view, img)
                shutil.copyfile(src, dst)

    def get_dicom_header(self, dicom_loc):
        # read dicom file and return header information and image
        ds = pydicom.read_file(dicom_loc, force=True)

        # get patient, study, and series information
        patient_id = clean_text(ds.get("PatientID", "NA"))
        series_description = clean_text(ds.get("SeriesDescription", "NA"))

        # generate new, standardized file name
        modality = ds.get("Modality","NA")
        series_instance_uid = ds.get("SeriesInstanceUID","NA")
        series_number = ds.get('SeriesNumber', 'NA')
        instance_number = int(ds.get("InstanceNumber","0"))
        image_position_patient = ds.get("ImagePositionPatient", 'NA')
        image_orientation_patient = ds.get("ImageOrientationPatient", 'NA')
        pixel_spacing = ds.get("PixelSpacing", 'NA')

        # load image data
        array = ds.pixel_array

        return patient_id, dicom_loc, modality, series_instance_uid, \
               series_number, instance_number, image_position_patient, image_orientation_patient, pixel_spacing, array, series_description
    
    def store_dicom_info(self):
        unsorted_list = []
        for root, dirs, files in os.walk(self.src):
            for file in files:
                if ".dcm" in file: 
                    unsorted_list.append(os.path.join(root, file))

        output = []
        for dicom_loc in unsorted_list:
            output.append(self.get_dicom_header(dicom_loc))

        # generated pandas dataframe to store information from headers
        self.df = pd.DataFrame(sorted(output), columns=['Patient ID',
                                            'Filename',
                                            'Modality',
                                            'Series ID',
                                            'Series Number',
                                            'Instance Number',
                                            'Image Position Patient',
                                            'Image Orientation Patient',
                                            'Pixel Spacing', 
                                            'Img',
                                            'Series Description'])

        self.merge_dicom_frames()
        
    def merge_dicom_frames(self):
        # Each series is a separate row in the dataframe
        # Merge frames for each series
        output = []
        unique_series = self.df['Series Number'].unique()

        # Sometimes, multiple series are stored together as one dicom 'series'. I heavily frown upon this practice. However, if it is the case, we need to split them up.
        # Let's try to find this by whether the image position patient changes between frames
        for series in unique_series:
            series_rows = self.df[self.df['Series Number'] == series]
            series_rows = series_rows.sort_values('Instance Number')

            all_img_positions = series_rows['Image Position Patient'].values
            same_position = [np.all(all_img_positions[i] == all_img_positions[0]) for i in range(len(all_img_positions))]
            if not np.all(same_position):
                # Find out how many series are merged
                num_merged_series = len(all_img_positions) // len(np.where(same_position)[0])
                idx_split = [len(all_img_positions) // num_merged_series * i for i in range(num_merged_series)]
                unique_image_positions = [all_img_positions[i] for i in idx_split]

                self.my_logger.info(f"Series {series} contains {num_merged_series} merged series. Splitting...")

                max_series_num = self.df['Series Number'].max()

                self.my_logger.info(f"New 'synthetic' series will range from: {max_series_num+1} to {max_series_num+num_merged_series}")
                
                for i in range(0,num_merged_series):
                    series_rows_split = series_rows[series_rows['Image Position Patient'] == unique_image_positions[i]]
                    series_rows_split = series_rows_split.sort_values('Instance Number')
                    img = np.stack(series_rows_split['Img'].values, axis=0)

                    num_phases = img.shape[0]

                    series_num = max_series_num + i + 1 # New series number ('fake' series number)

                    # Add to output
                    output.append([series_rows_split['Patient ID'].values[0], series_rows_split['Filename'].values[0], series_rows_split['Modality'].values[0], series_rows_split['Series ID'].values[0], series_num, series_rows_split['Image Position Patient'].values[0], series_rows_split['Image Orientation Patient'].values[0], series_rows_split['Pixel Spacing'].values[0], img, num_phases, series_rows_split['Series Description'].values[0]])

            else: # Just merge rows, no need to split series
                img = np.stack(series_rows['Img'].values, axis=0)

                num_phases = img.shape[0]

                # Add to output
                output.append([series_rows['Patient ID'].values[0], series_rows['Filename'].values[0], series_rows['Modality'].values[0], series_rows['Series ID'].values[0], series_rows['Series Number'].values[0], series_rows['Image Position Patient'].values[0], series_rows['Image Orientation Patient'].values[0], series_rows['Pixel Spacing'].values[0], img, num_phases, series_rows['Series Description'].values[0]])
 
        # generated pandas dataframe to store information from headers
        self.df = pd.DataFrame(sorted(output), columns=['Patient ID',
                                            'Filename',
                                            'Modality',
                                            'Series ID',
                                            'Series Number',
                                            'Image Position Patient',
                                            'Image Orientation Patient',
                                            'Pixel Spacing', 
                                            'Img',
                                            'Frames Per Slice',
                                            'Series Description'])
    
class ViewSelectorMetadata: # TODO: Merge with ViewSelector

    def __init__(self, src, dst, model_path, csv_path, my_logger):
        self.src = src
        self.dst = dst
        self.model = joblib.load(model_path)
        self.df = None
        self.unique_df = {}
        self.sorted_dict = {}
        self.csv_path = csv_path
        self.my_logger = my_logger

    def get_dicom_header(self, dicom_loc):

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

    def predict_views(self):

        unsorted_list = [os.path.join(root, file) for root, _, files in os.walk(self.src) for file in files if ".dcm" in file]

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
        self.predict()

        if len(self.unique_df) == 0:
            self.my_logger.warning(f'DataFrame is empty')

        if len(self.unique_df) > 0:
            output_path = os.path.join(self.dst, 'view-classification')
            self.unique_df.to_csv(f'{output_path}/metadata_view_predictions.csv', index=False)

    def predict(self):
        files = [os.path.join(root, file) for root, _, files in os.walk(os.path.join(self.dst, 'view-classification', 'temp')) for file in files if ".dcm" in file]

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

        sids = [n for n in os.listdir(os.path.join(self.dst,  'view-classification', 'temp'))]

        output_dataframe = []
        view_class_map = {'SA': 'SAX', '2CH LT': '2ch', '2CH RT': '2ch-RT', '3CH': '3ch', '4CH': '4ch', 'LVOT': 'LVOT', 'RVOT': 'RVOT', 'RVOT-T': 'RVOT-T', 'SAX-atria': 'SAX-atria', 'OTHER': 'OTHER'}
        for ids in sids:
            dcm = [n for n in os.listdir(os.path.join(self.dst, 'view-classification', 'temp', ids)) if 'dcm' in n]

            ds = pydicom.dcmread(os.path.join(self.dst, 'view-classification', 'temp', ids, dcm[0]))

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
            predicted_view = view_class_map[y_pred[0]]     

            output_dataframe.append([ds.SeriesNumber, predicted_view, 1, len(dcm)])

        # self.unique_df = pd.DataFrame(output_dataframe, columns=['Patient ID',
        #                                                         'Instance Number',
        #                                                         'Series InstanceUID',
        #                                                         'Series Number',
        #                                                         'Number of Frames',
        #                                                         'Predicted view'])

        self.unique_df = pd.DataFrame(output_dataframe, columns=['Series Number', 'Predicted View', 'Confidence', 'Frames Per Slice'])

        # delete temp folder
        shutil.rmtree(os.path.join(self.dst, 'view-classification', 'temp'))

    def sort_dicom_per_series(self):
        # Each series is a separate row in the dataframe
        # Merge frames for each series
        unique_series = self.df[['Series Number', 'Image Position Patient']].drop_duplicates()

        self.my_logger.info(f"{len(unique_series)} unique series found")
        count = 0

        os.makedirs(os.path.join(self.dst, 'view-classification', 'temp'), exist_ok=True)

        for _, row in unique_series.iterrows():

            series_rows = self.df.loc[(self.df['Series Number'] == row['Series Number']) & (self.df['Image Position Patient'] == row['Image Position Patient'])]

            if len(series_rows) < 10: # unlikely to be a cine
                self.my_logger.warning(f"Removing series {row['Series Number']} - less than 10 frames")
                continue

            series_rows = series_rows.sort_values('Trigger Time')
            pos = series_rows['Image Position Patient'].values[0]

            key = f'{series_rows["Series Number"].values[0]}_{pos[2]}'
            dcm_path = os.path.join(self.dst, 'view-classification', 'temp',key)
            os.makedirs(dcm_path, exist_ok=True) 

            count = 0
            for name in series_rows['Filename']:
                num = int(series_rows['Trigger Time'].values[count])
                shutil.copy(name, dcm_path / Path(f'{num:05}.dcm')) 
                count += 1

            self.sorted_dict[key] = series_rows
            

        
            
    