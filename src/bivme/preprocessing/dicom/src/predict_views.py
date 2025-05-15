import os
import sys
import shutil
import pydicom
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

from bivme.preprocessing.dicom.src.utils import from_2d_to_3d
from bivme.preprocessing.dicom.src.viewselection import ViewSelector

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


def predict_views(vs):
    if vs.type == 'metadata':
        predict_on_metadata(vs)
    elif vs.type == 'image':
        predict_on_images(vs)

def predict_on_metadata(vs):
    vs.prepare_data_for_prediction()

    if len(vs.df) == 0:
        vs.my_logger.error("No series found. This means that after excluding invalid series descriptions and images with less than 10 frames, this case has no eligible cine images. Please check your input directory.")
        sys.exit(0)

    view_class_map = {'SA': 'SAX', '2CH LT': '2ch', '2CH RT': '2ch-RT', '3CH': '3ch', '4CH': '4ch', 'LVOT': 'LVOT', 'RVOT': 'RVOT', 'RVOT-T': 'RVOT-T', 'SAX-atria': 'SAX-atria', 'OTHER': 'OTHER'}

    metadata_model_path = os.path.join(vs.model, "ViewSelection", "metadata-based_model.joblib")
    metadata_model = joblib.load(metadata_model_path)

    files = [os.path.join(root, file) for root, _, files in os.walk(os.path.join(vs.dst, 'view-classification', 'temp')) for file in files if ".dcm" in file]

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

    sids = [n for n in os.listdir(os.path.join(vs.dst,  'view-classification', 'temp'))]

    output_dataframe = []
    for ids in sids:
        dcm = [n for n in os.listdir(os.path.join(vs.dst, 'view-classification', 'temp', ids)) if 'dcm' in n]

        ds = pydicom.dcmread(os.path.join(vs.dst, 'view-classification', 'temp', ids, dcm[0]))

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
        
        scaler = metadata_model.scaler
        scaled_predictors = scaler.transform(predictors.reshape(1, -1))

        y_pred = metadata_model.predict(scaled_predictors)
        predicted_view = view_class_map[y_pred[0]]     

        series_num = ids.split('_')[0]
        output_dataframe.append([series_num, predicted_view, 1, len(dcm)])

    # Save to csv
    output_df = pd.DataFrame(output_dataframe, columns=['Series Number', 'Predicted View', 'Confidence', 'Frames Per Slice'])
    output_df.to_csv(vs.csv_path, mode='w', index=False)

    # delete temp folder
    shutil.rmtree(os.path.join(vs.dst, 'view-classification', 'temp'))

def predict_on_images(vs):
    vs.prepare_data_for_prediction()

    if len(vs.df) == 0:
        vs.my_logger.error("No series found. This means that after excluding invalid series descriptions and images with less than 10 frames, this case has no eligible cine images. Please check your input directory.")
        sys.exit(0)

    view_label_map = {'2ch': 0, '2ch-RT': 1, '3ch': 2, '4ch': 3, 'LVOT': 4, 
                'OTHER': 5, 'RVOT': 6, 'RVOT-T': 7, 'SAX': 8, 'SAX-atria': 9}
    
    test_annotations = os.path.join(vs.dst, 'view-classification', 'test_annotations.csv') # Dummy annotations file
    dir_img_test = os.path.join(vs.dst, 'view-classification', 'unsorted') # Directory of images to predict. Predictions are run on .pngs
    
    # Load model from file
    loaded_model_path = os.path.join(vs.model, "ViewSelection", "resnet50-v13.pth")
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
    
    # vs.my_logger.info("Running view predictions...")
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
            confidence = confidence.cpu().numpy().T
            # confidence = np.max(confidence, axis=1)

            new_row = pd.DataFrame({'image_name': img_names, 
                                    'predicted_label': predicted_labels, 
                                    'confidence_0': confidence[0],
                                    'confidence_1': confidence[1],
                                    'confidence_2': confidence[2],
                                    'confidence_3': confidence[3],
                                    'confidence_4': confidence[4],
                                    'confidence_5': confidence[5],
                                    'confidence_6': confidence[6],
                                    'confidence_7': confidence[7],
                                    'confidence_8': confidence[8],
                                    'confidence_9': confidence[9]})
            
            test_pred_df = pd.concat([test_pred_df, new_row], ignore_index=True)

    # Determine view class of each series
    output_df = pd.DataFrame(columns=['Series Number', 
                                      'Predicted View', 
                                        'Vote Share',
                                      'Frames Per Slice',
                                      f'{list(view_label_map.keys())[0]} confidence',
                                        f'{list(view_label_map.keys())[1]} confidence',
                                        f'{list(view_label_map.keys())[2]} confidence',
                                        f'{list(view_label_map.keys())[3]} confidence',
                                        f'{list(view_label_map.keys())[4]} confidence',
                                        f'{list(view_label_map.keys())[5]} confidence',
                                        f'{list(view_label_map.keys())[6]} confidence',
                                        f'{list(view_label_map.keys())[7]} confidence',
                                        f'{list(view_label_map.keys())[8]} confidence',
                                        f'{list(view_label_map.keys())[9]} confidence'])


    # Determine view class from majority vote across all frames
    for series in vs.df['Series Number'].unique():
        series_views = test_pred_df[test_pred_df['image_name'].str.startswith(f'{series}_')]

        view_counts = series_views['predicted_label'].value_counts()
        view_counts = view_counts / view_counts.sum()

        # Get most common view
        predicted_view = view_counts.idxmax()
        
        # Get mean confidences
        confidences = [series_views[f'confidence_{i}'].values for i in range(10)]
        confidences = np.mean(confidences, axis=1)

        new_row = pd.DataFrame({'Series Number': [series], 
                                'Predicted View': [list(view_label_map.keys())[predicted_view]], 
                                'Vote Share': [view_counts[predicted_view]],
                                'Frames Per Slice': [len(series_views)],
                                f'{list(view_label_map.keys())[0]} confidence': [confidences[0]],
                                f'{list(view_label_map.keys())[1]} confidence': [confidences[1]],
                                f'{list(view_label_map.keys())[2]} confidence': [confidences[2]],
                                f'{list(view_label_map.keys())[3]} confidence': [confidences[3]],
                                f'{list(view_label_map.keys())[4]} confidence': [confidences[4]],
                                f'{list(view_label_map.keys())[5]} confidence': [confidences[5]],
                                f'{list(view_label_map.keys())[6]} confidence': [confidences[6]],
                                f'{list(view_label_map.keys())[7]} confidence': [confidences[7]],
                                f'{list(view_label_map.keys())[8]} confidence': [confidences[8]],
                                f'{list(view_label_map.keys())[9]} confidence': [confidences[9]]})

        
        output_df = pd.concat([output_df, new_row], ignore_index=True)
    
    # Save to csv
    output_df.to_csv(vs.csv_path, mode='w', index=False)

    # Remove dummy annotations
    os.remove(test_annotations)