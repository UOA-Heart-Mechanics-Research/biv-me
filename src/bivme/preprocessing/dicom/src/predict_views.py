import os
import glob
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

from bivme.preprocessing.dicom.src.viewselection import ViewSelector
from bivme.preprocessing.dicom.src.guidepointprocessing import inverse_coordinate_transformation
from bivme.preprocessing.dicom.src.utils import plane_intersect


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

    # Create a plane for each series
    plane_dict = {}
    for idx, series_df in vs.df.iterrows():
        # Extract position, orientation, spacing, and image size
        position = series_df['Image Position Patient']
        orientation = series_df['Image Orientation Patient']
        spacing = series_df['Pixel Spacing']
        image_size = series_df['Img'].shape[1:3]

        # Create four cartesian points representing the corners of the image in 3D space
        origin = inverse_coordinate_transformation([0, 0], position, orientation, spacing)[:3]
        corner_1 = inverse_coordinate_transformation([image_size[0], 0], position, orientation, spacing)[:3]
        corner_2 = inverse_coordinate_transformation([0, image_size[1]], position, orientation, spacing)[:3]
        corner_3 = inverse_coordinate_transformation([image_size[0], image_size[1]], position, orientation, spacing)[:3]

        # Convert to Ax + By + Cz + D = 0 form
        A = (corner_1[1] - origin[1]) * (corner_2[2] - origin[2]) - (corner_1[2] - origin[2]) * (corner_2[1] - origin[1])
        B = (corner_1[2] - origin[2]) * (corner_2[0] - origin[0]) - (corner_1[0] - origin[0]) * (corner_2[2] - origin[2])
        C = (corner_1[0] - origin[0]) * (corner_2[1] - origin[1]) - (corner_1[1] - origin[1]) * (corner_2[0] - origin[0])
        D = -(A * origin[0] + B * origin[1] + C * origin[2])

        plane_dict[series_df['Series Number']] = {
            'origin': origin,
            'corner_1': corner_1,
            'corner_2': corner_2,
            'corner_3': corner_3,
            'A': A,
            'B': B,
            'C': C,
            'D': D,
        }

    # Find if planes intersect
    intersection_dict = {}
    intersection_range = 200

    for series_num_a, plane_a in plane_dict.items():
        for series_num_b, plane_b in plane_dict.items():
            if series_num_a == series_num_b:
                continue
            if (series_num_a, series_num_b) in intersection_dict or (series_num_b, series_num_a) in intersection_dict:
                continue
            
            centroid_a = np.mean([plane_a['origin'], plane_a['corner_1'], plane_a['corner_2'], plane_a['corner_3']], axis=0)
            centroid_b = np.mean([plane_b['origin'], plane_b['corner_1'], plane_b['corner_2'], plane_b['corner_3']], axis=0)

            point_a, point_b = plane_intersect((plane_a['A'], plane_a['B'], plane_a['C'], plane_a['D']),
                                                (plane_b['A'], plane_b['B'], plane_b['C'], plane_b['D']))
            
            # Check if the intersection points are valid (not NaN)
            if np.isnan(point_a).any() or np.isnan(point_b).any():
                intersection_dict[(series_num_a, series_num_b)] = None
            elif np.linalg.norm(point_a - point_b) < 1e-3:
                # If the intersection points are the same, skip this pair
                intersection_dict[(series_num_a, series_num_b)] = None
            elif np.linalg.norm(point_a - centroid_a) > intersection_range and np.linalg.norm(point_b - centroid_b) > intersection_range:
                # If the intersection points are too far from the centroids of their respective series, skip this pair
                intersection_dict[(series_num_a, series_num_b)] = None
            else:
                intersection_dict[(series_num_a, series_num_b)] = (point_a, point_b)
    
    # Simplify
    simplified_intersection_dict = {}
    for (series_num_a, series_num_b), points in intersection_dict.items():
        if points is None:
            continue
        
        if series_num_a not in simplified_intersection_dict:
            simplified_intersection_dict[series_num_a] = [series_num_b]
        else:
            simplified_intersection_dict[series_num_a].append(series_num_b)

        # And then add the reverse
        if series_num_b not in simplified_intersection_dict:
            simplified_intersection_dict[series_num_b] = [series_num_a]
        else:
            simplified_intersection_dict[series_num_b].append(series_num_a)
    
    # Cluster into groups that intersect with the same slices
    groups = []
    all_series_nums = set(simplified_intersection_dict.keys())

    for series_num, intersecting_series in simplified_intersection_dict.items():
        group = []

        for s in all_series_nums:
            if s not in intersecting_series:
                group.append(s)

        groups.append(sorted(group))
    
    group_sets = set(tuple(group) for group in groups)
    if not group_sets:
        vs.my_logger.warning("No intersecting series found, which is unexpected. Either metadata view prediction failed, or you should check your input data - it might be only SAX images.")
        largest_set = None
        series_nums = vs.df['Series Number'].unique()
        simplified_intersection_dict = {series: [] for series in series_nums}
    else:
        largest_set = max(group_sets, key=len)

    view_predictions = []

    for series, intersecting_series in simplified_intersection_dict.items():
        if largest_set is None:
            view_predictions.append([series, 'Other'])

        elif series in largest_set:     # Largest set of non-intersecting series assumed to be SAX
            view_predictions.append([series, 'SAX'])
        elif set(largest_set).issubset(set(intersecting_series)):   # LAX views will be the ones that intersect with every SAX view
            view_predictions.append([series, 'LAX'])
        else:
            view_predictions.append([series, 'Other'])

    # Save to csv
    output_df = pd.DataFrame(view_predictions, columns=['Series Number', 'View Class'])
    output_df.to_csv(vs.csv_path, mode='w', index=False)


def predict_on_images(vs):
    vs.prepare_data_for_prediction()

    if len(vs.df) == 0:
        vs.my_logger.error("No series found. This means that after excluding invalid series descriptions and images with less than 10 frames, this case has no eligible cine images. Please check your input directory.")
        sys.exit(0)

    old_view_label_map = {'2ch': 0, '2ch-RT': 1, '3ch': 2, '4ch': 3, 'LVOT': 4, 
                'OTHER': 5, 'RVOT': 6, 'RVOT-T': 7, 'SAX': 8, 'SAX-atria': 9}
    
    view_label_map = {'2ch': 0, '2ch-RV': 1, '3ch': 2, '4ch': 3, 'LVOT': 4, 
            'SAX-other': 5, 'RVOT': 6, 'RVOT-oblique': 7, 'SAX': 8, 'SAX-atria': 9}
    
    test_annotations = os.path.join(vs.dst, 'view-classification', 'test_annotations.csv') # Dummy annotations file
    dir_img_test = os.path.join(vs.dst, 'view-classification', 'unsorted') # Directory of images to predict. Predictions are run on .pngs
    
    # Load model from file
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    try:
        loaded_model_path = glob.glob(os.path.join(vs.model, "ViewSelection") + "/resnet50-v32.pth")[0]
    except IndexError:
        vs.my_logger.error("No image view selection model found. Make sure you followed the installation instructions for installing the deep learning models.")
        sys.exit(0)

    loaded_model = torchvision.models.resnet50()
    loaded_model.fc = nn.Linear(2048, 10)

    # Load model
    loaded_model.load_state_dict(torch.load(loaded_model_path, map_location=device))

    model = loaded_model

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
    batch_size = len(test_dataset)//4
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # vs.my_logger.info("Running view predictions...")
    model.eval()
    model.to(device)

    test_pred_df = pd.DataFrame(columns=['image_name', 'predicted_label'])

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            # Add to dataframe
            predicted_labels = predicted.cpu().numpy()

            img_names = test_dataset.img_labels['image_name'].values[batch_size*i:batch_size*(i+1)]

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
                                      'Series Description',
                                      'Slice Location',
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

        # Get series description and slice location
        series_description = vs.df[vs.df['Series Number'] == series]['Series Description'].values[0]
        slice_location = vs.df[vs.df['Series Number'] == series]['Slice Location'].values[0]

        new_row = pd.DataFrame({'Series Number': [series], 
                                'Series Description': [series_description],
                                'Slice Location': [slice_location],
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