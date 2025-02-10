import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

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
    vs.prepare_data_for_prediction()

    view_label_map = {'2ch': 0, '2ch-RT': 1, '3ch': 2, '4ch': 3, 'LVOT': 4, 
                'OTHER': 5, 'RVOT': 6, 'RVOT-T': 7, 'SAX': 8, 'SAX-atria': 9}
    
    test_annotations = os.path.join(vs.dst, 'view-classification', 'test_annotations.csv') # Dummy annotations file
    dir_img_test = os.path.join(vs.dst, 'view-classification', 'unsorted') # Directory of images to predict. Predictions are run on .pngs
    
    # Load model from file
    loaded_model_path = os.path.join(vs.model, "ViewSelection", "resnet50-v9.pth")
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
    
    vs.my_logger.info("Running view predictions...")
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

    # Determine view class from majority vote across all frames
    for series in vs.df['Series Number'].unique():
        series_views = test_pred_df[test_pred_df['image_name'].str.startswith(f'{series}_')]

        view_counts = series_views['predicted_label'].value_counts()
        view_counts = view_counts / view_counts.sum()

        # Get most common view
        predicted_view = view_counts.idxmax()
        confidence = view_counts.max() # TODO: This isn't really 'confidence' but the proportion of frames with this view. Change to 'Unanimity' or something

        new_row = pd.DataFrame({'Series Number': [series], 'Predicted View': [list(view_label_map.keys())[predicted_view]], 'Confidence': [confidence], 'Frames Per Slice': [len(series_views)]})
        output_df = pd.concat([output_df, new_row], ignore_index=True)
    
    # Save to csv
    output_df.to_csv(vs.csv_path, mode='w', index=False)

    # Remove dummy annotations
    os.remove(test_annotations)