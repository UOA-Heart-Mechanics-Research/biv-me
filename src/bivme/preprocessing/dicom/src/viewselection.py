import os
import shutil
import pydicom
import numpy as np
import pandas as pd
import PIL

def clean_text(string):

    # clean and standardize text descriptions, which makes searching files easier

    forbidden_symbols = ["*", ".", ",", "\"", "\\", "/", "|", "[", "]", ":", ";", " "]
    for symbol in forbidden_symbols:
        string = string.replace(symbol, "")  # replace all bad symbols

    return string.lower()

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
                img_data = PIL.Image.fromarray(img_data)
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
    




        
            
    