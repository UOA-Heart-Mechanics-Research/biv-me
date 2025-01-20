import os
import shutil
import pydicom
import numpy as np
import pandas as pd
import PIL
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import numpy as np
import os
import pandas as pd
import PIL

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def clean_text(string):

    # clean and standardize text descriptions, which makes searching files easier

    forbidden_symbols = ["*", ".", ",", "\"", "\\", "/", "|", "[", "]", ":", ";", " "]
    for symbol in forbidden_symbols:
        string = string.replace(symbol, "")  # replace all bad symbols

    return string.lower()

class ViewSelector:

    def __init__(self, src, dst, model, csv_path):
        self.src = src
        self.dst = dst
        self.model = model
        self.csv_path = csv_path
        self.df = None

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

        print(f"Data prepared for view prediction. {len(self.df)} image series found.")

    def predict_views(self):
        self.prepare_data_for_prediction()

        view_label_map = {'2ch': 0, '2ch-RT': 1, '3ch': 2, '4ch': 3, 'LVOT': 4, 
                    'OTHER': 5, 'RVOT': 6, 'RVOT-T': 7, 'SAX': 8, 'SAX-atria': 9}
        
        test_annotations = os.path.join(self.dst, 'view-classification', 'test_annotations.csv') # Dummy annotations file
        dir_img_test = os.path.join(self.dst, 'view-classification', 'unsorted') # Directory of images to predict. Predictions are run on .pngs
        
        # Load model from file
        loaded_model_path = os.path.join(self.model, "ViewSelection", "resnet50-v9.pth")
        loaded_model = torchvision.models.resnet50()
        loaded_model.fc = nn.Linear(2048, 10)
        loaded_model.load_state_dict(torch.load(loaded_model_path))

        model = loaded_model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get transforms
        orig_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # Normalise
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        test_dataset = CustomImageDataset(test_annotations, dir_img_test, transform=orig_transform)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        
        print("Running view predictions...")
        model.eval()
        model.to(device)

        test_pred_df = pd.DataFrame(columns=['image_name', 'predicted_label'])

        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                _, predicted = torch.max(outputs.data, 1)

                # Add to dataframe
                predicted_labels = predicted.cpu().numpy()

                img_names = test_dataset.img_labels['image_name'].values

                # Calculate confidence
                confidence = nn.functional.softmax(outputs, dim=1)
                confidence = confidence.cpu().numpy()
                confidence = np.max(confidence, axis=1)

                new_row = pd.DataFrame({'image_name': img_names, 'predicted_label': predicted_labels, 'confidence': confidence})
                test_pred_df = pd.concat([test_pred_df, new_row], ignore_index=True)


        # Determine view class of each series
        output_df = pd.DataFrame(columns=['Series Number', 'Predicted View', 'Confidence', 'Frames Per Slice'])

        # # Determine view class from first frame (frame 0)
        # for series in self.df['Series Number'].unique():
        #     series_views = test_pred_df[test_pred_df['image_name'].str.contains(f'{series}_0.png')]
        #     predicted_view = series_views['predicted_label'].values[0]
        #     confidence = series_views['confidence'].values[0]

        #     num_frames = len(self.df[self.df['Series Number'] == series])

        #     new_row = pd.DataFrame({'Series Number': [series], 'Predicted View': [list(view_label_map.keys())[predicted_view]], 'Confidence': [confidence], 'Frames Per Slice': [num_frames]})
        #     output_df = pd.concat([output_df, new_row], ignore_index=True)

        # Determine view class from majority vote across all frames
        for series in self.df['Series Number'].unique():
            series_views = test_pred_df[test_pred_df['image_name'].str.startswith(f'{series}_')]

            view_counts = series_views['predicted_label'].value_counts()
            view_counts = view_counts / view_counts.sum()

            # Get most common view
            predicted_view = view_counts.idxmax()
            confidence = view_counts.max() # TODO: This isn't really 'confidence' but the proportion of frames with this view. Change to 'Unanimity' or something

            new_row = pd.DataFrame({'Series Number': [series], 'Predicted View': [list(view_label_map.keys())[predicted_view]], 'Confidence': [confidence], 'Frames Per Slice': [len(series_views)]})
            output_df = pd.concat([output_df, new_row], ignore_index=True)
        
        # Save to csv
        output_df.to_csv(self.csv_path, mode='w', index=False)

        # Remove dummy annotations
        os.remove(test_annotations)

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
        for series in unique_series:
            series_rows = self.df[self.df['Series Number'] == series]
            # Order by instance number
            series_rows = series_rows.sort_values('Instance Number')
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


        
            
    