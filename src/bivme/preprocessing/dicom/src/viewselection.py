import os
import shutil
import pydicom
import numpy as np
import pandas as pd
import PIL.Image as Image
import cv2
import nibabel as nib


OLD_VIEW_NAMES = ['SAX-atria', 'SAX', 'OTHER', '2ch', '3ch', '4ch', 'RVOT', 'LVOT', '2ch-RT', 'RVOT-T', 'Excluded']
VIEW_NAMES = ['SAX-atria', 'SAX', 'SAX-other', '2ch', '3ch', '4ch', 'RVOT', 'LVOT', '2ch-RV', 'RVOT-oblique', 'Excluded']

class ViewSelector:
    def __init__(self, src, dst, model, type, csv_path, show_warnings, my_logger):
        self.src = src
        self.dst = dst
        self.model = model
        self.type = type
        self.df = None
        self.unique_df = {}
        self.sorted_dict = {}
        self.csv_path = csv_path
        self.my_logger = my_logger
        self.show_warnings = show_warnings
        self.excluded_df = None

    def load_predictions(self):
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

        if self.type == "image":
            for i, row in self.df.iterrows():
                img = row['Img']
                pixel_spacing = row['Pixel Spacing']
                for frame in range(img.shape[0]):
                    img_data = img[frame, :, :]

                    img_size = img_data.shape
                    # Minimum dimension should be 256
                    min_dim = min(img_size)
                    if min_dim != 256:
                        ratio = 256 / min_dim
                        new_size = (int(img_size[1] * ratio), int(img_size[0] * ratio))
                        min_dim = min(new_size)
                        if min_dim != 256:
                            new_size = (max(new_size[0], 256), max(new_size[1], 256)) # prevent rounding errors

                    else:
                        new_size = img_size

                    img_data = cv2.resize(img_data, new_size, interpolation=cv2.INTER_CUBIC)

                    # Crop to central square
                    original_shape = img_data.shape

                    if original_shape[0] != original_shape[1]:
                        # Crop to square
                        min_dim = min(original_shape[0], original_shape[1])
                        start_x = (original_shape[0] - min_dim) // 2
                        start_y = (original_shape[1] - min_dim) // 2

                        img_data = img_data[start_x:start_x+min_dim, start_y:start_y+min_dim]
                        original_shape = img_data.shape

                    # Apply clahe to img_data
                    min_value = min(0, np.min(img_data))
                    img_data += abs(min_value) # Shift to make all values positive because CLAHE requires uint type (0-)

                    
                    img_data = img_data.astype(np.uint16)

                    clahe = cv2.createCLAHE(tileGridSize=(1,1))
                    img_data_clahe = clahe.apply(img_data)
                    img_data = img_data_clahe

                    # 0-1 normalization
                    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
                    img_data = (img_data * 255).astype(np.uint8)

                    # Stack to 3 channels
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

        # self.my_logger.info(f"Data prepared for view prediction. {len(self.df)} image series found.")

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

            # Compatibility with older versions of view classification
            if view in OLD_VIEW_NAMES and view not in VIEW_NAMES:
                view = VIEW_NAMES[OLD_VIEW_NAMES.index(view)] # Map old view names to new ones

            os.makedirs(os.path.join(dir_sorted, view), exist_ok=True)
            
            # Grab images
            series_images = [f for f in os.listdir(dir_unsorted) if f.startswith(f'{series_number}_')]
            for img in series_images:
                src = os.path.join(dir_unsorted, img)
                dst = os.path.join(dir_sorted, view, img)
                shutil.copyfile(src, dst)

    def get_dicom_header(self, dicom_loc, array, idx=None):
        # read dicom file and return header information and image
        ds = pydicom.dcmread(dicom_loc, force=True, stop_before_pixels=True)
        # get patient, study, and series information
        patient_id = ds.get("PatientID", "NA")
        modality = ds.get("Modality","NA")

        try:
            instance_number = int(ds.get("InstanceNumber","NA"))
        except ValueError:
            if self.show_warnings:
                self.my_logger.warning(f"InstanceNumber for {dicom_loc} is not an integer.")
            instance_number = ds.get("InstanceNumber","NA")

        series_instance_uid = ds.get("SeriesInstanceUID","NA")

        try:
            series_number = int(ds.get('SeriesNumber', 'NA'))
        except ValueError:
            if self.show_warnings:
                self.my_logger.warning(f"SeriesNumber for {dicom_loc} is not an integer.")
            series_number = ds.get('SeriesNumber', 'NA')

        if idx is None:
            image_position_patient = ds.get("ImagePositionPatient", 'NA')
            image_orientation_patient = ds.get("ImageOrientationPatient", 'NA')
            pixel_spacing = ds.get("PixelSpacing", 'NA')

            echo_time = ds.get("EchoTime", 'NA')
            repetition_time = ds.get("RepetitionTime", 'NA')

            try:
                trigger_time = float(ds.get('TriggerTime', 'NA'))
            except ValueError:
                trigger_time = ds.get('TriggerTime', 'NA')

            encoding_direction = ds.get("InPlanePhaseEncodingDirection", 'NA')
            image_dimension = [ds.get('Rows', 'NA'), ds.get('Columns', 'NA')]
            slice_thickness = ds.get('SliceThickness', 'NA')
            slice_location = ds.get('SliceLocation', 'NA')
            series_description = ds.get('SeriesDescription', 'NA')


        else:
            frame_sequence = ds.get("PerFrameFunctionalGroupsSequence", None)
            if frame_sequence is None:
                if self.show_warnings:
                    self.my_logger.warning(f"Dicom file {dicom_loc} does not contain a PerFrameFunctionalGroupsSequence. Cannot extract frame {idx}.")
                return None
            frame_data = frame_sequence[idx]

            image_position_patient = frame_data.PlanePositionSequence[0].get("ImagePositionPatient", 'NA')
            image_orientation_patient = frame_data.PlaneOrientationSequence[0].get("ImageOrientationPatient", 'NA')
            pixel_spacing = frame_data.PixelMeasuresSequence[0].get("PixelSpacing", 'NA')

            mr_echo_sequence = frame_data.get("MREchoSequence", None)

            if mr_echo_sequence is not None:
                echo_time = mr_echo_sequence[0].get("EffectiveEchoTime", 'NA')

            else:
                echo_time = ds.get("EchoTime", 'NA')

            shared_functional_groups_sequence = ds.get("SharedFunctionalGroupsSequence", None)
            if shared_functional_groups_sequence is not None:
                mrfov_geometry_sequence = shared_functional_groups_sequence[0].get("MRFOVGeometrySequence", None)
                if mrfov_geometry_sequence is not None:
                    encoding_direction = mrfov_geometry_sequence[0].get("InPlanePhaseEncodingDirection", 'NA')
                else:
                    encoding_direction = ds.get("InPlanePhaseEncodingDirection", 'NA')
                try:
                    repetition_time = shared_functional_groups_sequence[0][0x0018,0x9112][0].get("RepetitionTime", 'NA')
                except KeyError:
                    repetition_time = shared_functional_groups_sequence[0].get("RepetitionTime", 'NA')
            else:
                repetition_time = ds.get("RepetitionTime", 'NA')
                encoding_direction = ds.get("InPlanePhaseEncodingDirection", 'NA')

            cardiac_sync_sequence = frame_data.get("CardiacSynchronizationSequence", None)
            if cardiac_sync_sequence is not None:
                try:
                    trigger_time = float(cardiac_sync_sequence[0].get("NominalCardiacTriggerDelayTime", 'NA'))
                except ValueError:
                    trigger_time = cardiac_sync_sequence[0].get("NominalCardiacTriggerDelayTime", 'NA')
            else:
                trigger_time = ds.get('TriggerTime', 'NA')

            image_dimension = [ds.get('Rows', 'NA'), ds.get('Columns', 'NA')]

            slice_thickness = frame_data.PixelMeasuresSequence[0].get('SliceThickness', 'NA')

            try:
                slice_location = frame_data[0x0021, 0x11FE][0][0x0021,0x1188].value
            except KeyError:
                slice_location = ds.get('SliceLocation', 'NA')

            series_description = ds.get('SeriesDescription', 'NA')

            instance_number = idx + 1

        return patient_id, dicom_loc, modality, instance_number, series_instance_uid, series_number, tuple(image_position_patient), image_orientation_patient, pixel_spacing, echo_time, repetition_time, trigger_time, encoding_direction, image_dimension, slice_thickness, array, slice_location, series_description
    
    def store_dicom_info(self):
        unsorted_list = []
        for root, dirs, files in os.walk(self.src):
            for file in files:
                if ".dcm" in file:
                    unsorted_list.append(os.path.join(root, file))
        output = []
        for dicom_loc in unsorted_list:
            try:
                ds = pydicom.dcmread(dicom_loc, force=True)
                array = ds.pixel_array.astype(np.float32)

                if len(array.shape) == 2:
                    mdata = self.get_dicom_header(dicom_loc, array)
                    if mdata is not None:
                        output.append(mdata)
                elif len(array.shape) == 3:
                    # Multi-frame dicom, split into individual frames
                    for frame in range(array.shape[0]):
                        array_frame = array[frame, :, :]
                        mdata = self.get_dicom_header(dicom_loc, array_frame, idx=frame)
                        if mdata is not None:
                            output.append(mdata)
                else:
                    raise ValueError(f"Unexpected number of dimensions in dicom image: {len(array.shape)}")
            except:
                if self.show_warnings:
                    self.my_logger.warning(f"Could not read {dicom_loc}. Image or metadata may be incompatible.")
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
                                                'Encoding Direction',
                                                'Image Dimension',
                                                'Slice Thickness',
                                                'Img',
                                                'Slice Location',
                                                'Series Description'])
        self.sort_dicom_per_series()
        
    def sort_dicom_per_series(self):
        # Each series is a separate row in the dataframe
        # Merge frames for each series
        output = []
        unique_series = self.df[['Series Number']].drop_duplicates()

        max_series_num = self.df['Series Number'].max()

        for _, row in unique_series.iterrows():
            series = row['Series Number']
            series_rows = self.df.loc[(self.df['Series Number'] == series)]

            if len(series_rows) < 10: # unlikely to be a cine
                if self.show_warnings:
                    self.my_logger.warning(f"Removing series {series} - less than 10 frames")
                continue

            all_img_positions = series_rows['Image Position Patient'].values
            same_position = [np.all(all_img_positions[i] == all_img_positions[0]) for i in range(len(all_img_positions))]

            # get basic dicom information
            patient_id = series_rows['Patient ID'].values[0]
            filename = series_rows['Filename'].values[0]
            modality = series_rows['Modality'].values[0]
            series_instance_uid = series_rows['Series InstanceUID'].values[0]
            series_description = series_rows['Series Description'].values[0]

            if not np.all(same_position):
                # Sort the merged series in a consistent order based on Slice Location and Instance Number (helps when interacting with GUI)
                all_slice_locations = series_rows['Slice Location'].values
                all_instance_numbers = series_rows['Instance Number'].values

                try:
                    all_slice_locations = np.array([float(loc) for loc in all_slice_locations])
                except ValueError:
                    all_slice_locations = np.array([0] * len(all_img_positions))

                try:
                    all_instance_numbers = np.array([float(loc) for loc in all_instance_numbers])
                except ValueError:
                    all_instance_numbers = np.array([0] * len(all_img_positions))

                if all_slice_locations is None and all_instance_numbers is None:
                    if self.show_warnings:
                        self.my_logger.warning(f"Could not sort series {series} due to missing slice location and instance number fields. Skipping...")
                    continue

                unsorted = [(pos, slice_loc, inst_num) for pos, slice_loc, inst_num in zip(all_img_positions, all_slice_locations, all_instance_numbers)]
                sorted_unsorted = sorted(unsorted, key=lambda x: (x[1], x[2]))  # Sort by slice location, then instance number
                all_img_positions = [item[0] for item in sorted_unsorted]

                unique_image_positions = []
                for pos in all_img_positions:
                    if pos not in unique_image_positions:
                        unique_image_positions.append(pos)

                # Find how many unique image positions there are
                num_merged_series = len(unique_image_positions)

                if self.show_warnings:
                    self.my_logger.info(f"Series {series} contains {num_merged_series} merged series. Splitting...")
                    self.my_logger.info(f"New 'synthetic' series will range from: {max_series_num+1} to {max_series_num+num_merged_series}")
                
                for i in range(0,num_merged_series):
                    series_rows_split = series_rows[series_rows['Image Position Patient'] == unique_image_positions[i]]


                    try:
                        if isinstance(series_rows_split['Trigger Time'].values[0], str):
                            series_rows_split = series_rows_split.sort_values('Instance Number')
                        else:
                            series_rows_split = series_rows_split.sort_values('Trigger Time')
                    except Exception as e:
                        # Keep the original order if sorting fails
                        if self.show_warnings:
                            self.my_logger.warning(f"Could not sort series {series} split {i+1}. Keeping original order. Error: {e}")

                    series_num = max_series_num + i + 1 # New series number ('fake' series number)

                    if len(series_rows_split) < 10: # unlikely to be a cine
                        if self.show_warnings:
                            self.my_logger.warning(f"Removing series {series_num} - less than 10 frames")
                        continue
                    
                    try:
                        img = np.stack(series_rows_split['Img'].values, axis=0)
                    except ValueError:
                        if self.show_warnings:
                            self.my_logger.warning(f"Could not stack images for series {series_num}. Skipping...")
                        continue

                    num_phases = img.shape[0]

                    # slice specific dicom information
                    image_position_patient = series_rows_split['Image Position Patient'].values[0]
                    image_orientation_patient = series_rows_split['Image Orientation Patient'].values[0]
                    pixel_spacing = series_rows_split['Pixel Spacing'].values[0]
                    slice_location = series_rows_split['Slice Location'].values[0]
                    encoding_direction = series_rows_split['Encoding Direction'].values[0]

                    if type(pixel_spacing) is str or type(image_position_patient) is str or type(image_orientation_patient) is str:
                        if self.show_warnings:
                            self.my_logger.warning(f"Invalid cartesian transform metadata for series {series_num}. Skipping...")
                        continue

                    # Add to output
                    output.append([patient_id, filename, modality, series_instance_uid, series_num, image_position_patient, image_orientation_patient, pixel_spacing, 
                                   slice_location, encoding_direction, img, num_phases, series_description])
                        
                # Update max series number
                max_series_num += num_merged_series

            else:
                try:
                    if isinstance(series_rows['Trigger Time'].values[0], str):
                        series_rows = series_rows.sort_values('Instance Number')
                    else:
                        series_rows = series_rows.sort_values('Trigger Time')
                except Exception as e:
                    # Keep the original order if sorting fails
                    if self.show_warnings:
                        self.my_logger.warning(f"Could not sort series {series}. Keeping original order. Error: {e}")

                try:
                    img = np.stack(series_rows['Img'].values, axis=0)
                except ValueError:
                    if self.show_warnings:
                        self.my_logger.warning(f"Could not stack images for series {series}. Skipping...")
                    continue

                num_phases = img.shape[0]

                # slice specific dicom information
                image_position_patient = series_rows['Image Position Patient'].values[0]
                image_orientation_patient = series_rows['Image Orientation Patient'].values[0]
                pixel_spacing = series_rows['Pixel Spacing'].values[0]
                slice_location = series_rows['Slice Location'].values[0]
                encoding_direction = series_rows['Encoding Direction'].values[0]

                if type(pixel_spacing) is str or type(image_position_patient) is str or type(image_orientation_patient) is str:
                    if self.show_warnings:
                        self.my_logger.warning(f"Invalid cartesian transform metadata for series {series}. Skipping...")
                    continue

                # Add to output
                output.append([patient_id, filename, modality, series_instance_uid, series, image_position_patient, image_orientation_patient, pixel_spacing, 
                               slice_location, encoding_direction, img, num_phases, series_description])

        # generated pandas dataframe to store information from headers
        self.df = pd.DataFrame(sorted(output), columns=['Patient ID',
                                            'Filename',
                                            'Modality',
                                            'Series InstanceUID',
                                            'Series Number',
                                            'Image Position Patient',
                                            'Image Orientation Patient',
                                            'Pixel Spacing', 
                                            'Slice Location',
                                            'Encoding Direction',
                                            'Img',
                                            'Frames Per Slice',
                                            'Series Description'])