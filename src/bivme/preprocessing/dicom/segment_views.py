import os
import numpy as np
import nibabel as nib
import shutil
import torch

# Set nnUNet environment variables so it doesn't scream at you with warnings
os.environ['nnUNet_raw'] = '.'
os.environ['nnUNet_preprocessed'] = '.'
os.environ['nnUNet_results'] = '.'

import nnunetv2 as nnunetv2
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def init_nnUNetv2(model_folder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predictor = nnUNetPredictor(tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    return predictor

def write_nifti(slice_id, pixel_array, pixel_spacing, input_folder, view, version):
    if version == '2d':
        for frame, img in enumerate(pixel_array):
            img = img.astype(np.float32)
            # Transpose
            img = np.transpose(img)

            affine = np.eye(4)
            affine[0, 0] = pixel_spacing[0]
            affine[1, 1] = pixel_spacing[1]
            img_nii = nib.Nifti1Image(img, affine)
            nib.save(img_nii, os.path.join(input_folder, view, '{}_2d_{}_{:03}_0000.nii.gz'.format(view, slice_id, frame)))
            
    elif version == '3d':
        img = pixel_array.astype(np.float32)
        # Transpose so that the last dimension is the number of frames
        img = np.transpose(img, (1, 2, 0))
        # Transpose width and height
        img = np.transpose(img, (1, 0, 2))

        affine = np.eye(4)
        affine[0, 0] = pixel_spacing[0]
        affine[1, 1] = pixel_spacing[1]
        img_nii = nib.Nifti1Image(img, affine)
        nib.save(img_nii, os.path.join(input_folder, view, '{}_3d_{}_0000.nii.gz'.format(view, slice_id)))

def predict_view(input_folder, output_folder, model, view, version, dataset):
    print('*** Making predictions for {} images ***'.format(view))

    # Define the trained model to use (Specified by the Task)
    if version == '2d':
        model_folder_name = os.path.join(model,"Segmentation/{}/nnUNetTrainer__nnUNetPlans__2d/".format(dataset))
    elif version == '3d':
        model_folder_name = os.path.join(model,"Segmentation/{}/nnUNetTrainer__nnUNetPlans__3d_fullres/".format(dataset))

    view_input_folder = os.path.join(input_folder, view)
    view_output_folder = os.path.join(output_folder, view)
        
    if len(os.listdir(view_input_folder)) > 0:

        # Initialize nnUNet model
        predictor = init_nnUNetv2(model_folder_name)

        # Make predictions
        predictor.predict_from_files(
            view_input_folder,
            view_output_folder,
            save_probabilities=False, overwrite=True, num_processes_preprocessing=2, 
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0
        )

        print('Done with {}\n'.format(view))

    if version == '2d':
        # Concatenate 2D predictions to 3D
        segmentations = [f for f in os.listdir(view_output_folder) if f.endswith('.nii.gz')]
        slices = np.unique([f.split('_')[-2] for f in segmentations])
        for slice in slices:
            segs = [f for f in segmentations if slice==f.split('_')[-2]]
            # Sort by frame number
            segs.sort(key=lambda x: int(x.split('_')[-1].replace('.nii.gz','')))

            concat_seg = []
            for seg in segs:
                img = nib.load(os.path.join(view_output_folder, seg))
                img = img.get_fdata()
                concat_seg.append(img)

            concat_seg = np.stack(concat_seg, axis=-1)
            # Save as 3D nii
            affine = np.eye(4)
            img_nii = nib.Nifti1Image(concat_seg, affine)
            nib.save(img_nii, os.path.join(view_output_folder, '{}_3d_{}.nii.gz'.format(view, slice)))

    elif version == '3d':
        # Split 3D predictions to 2D
        segmentations = [f for f in os.listdir(view_output_folder) if f.endswith('.nii.gz')]
        for seg in segmentations:
            img = nib.load(os.path.join(view_output_folder, seg))
            img = img.get_fdata()
            for frame in range(img.shape[-1]):
                img_frame = img[:, :, frame]
                affine = np.eye(4)
                img_nii = nib.Nifti1Image(img_frame, affine)
                nib.save(img_nii, os.path.join(view_output_folder, '{}_2d_{}_{:03}.nii.gz'.format(view, seg.split('_')[-1].replace('.nii.gz',''), frame)))
    

def segment_views(case, dst, model, slice_info_df, version):
    # define I/O parameters for nnUnet segmentation
    input_folder = os.path.join(dst, 'images')
    output_folder = os.path.join(dst, 'segmentations')

    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    else:
        shutil.rmtree(input_folder)
        os.makedirs(input_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        

    # nnunet models / tasks
    datasets_2d = ["Dataset210_SAX_2D", "Dataset211_2ch_2D", "Dataset212_3ch_2D", "Dataset213_4ch_2D", "Dataset214_RVOT_2D"]
    datasets_3d = ["Dataset220_SAX_3D", "Dataset221_2ch_3D", "Dataset222_3ch_3D", "Dataset223_4ch_3D", "Dataset224_RVOT_3D"]

    views = ['SAX', '2ch', '3ch', '4ch', 'RVOT']

    for i, view in enumerate(views):
        os.makedirs(os.path.join(input_folder, view), exist_ok=True)
        os.makedirs(os.path.join(output_folder, view), exist_ok=True)
        print(f'Writing {view} images to nifti files...\n')

        view_rows = slice_info_df[slice_info_df['View'] == view]
        for j, row in view_rows.iterrows():
            slice_id = row['Slice ID']
            pixel_array = row['Img']
            pixel_spacing = row['Pixel Spacing']
            write_nifti(slice_id, pixel_array, pixel_spacing, input_folder, view, version)

        print(f'Segmenting {view} images...\n')
        
        if version == '2d':
            dataset = datasets_2d[i]
            predict_view(input_folder, output_folder, model, view, version, dataset)
        elif version == '3d':
            dataset = datasets_3d[i]
            predict_view(input_folder, output_folder, model, view, version, dataset)
