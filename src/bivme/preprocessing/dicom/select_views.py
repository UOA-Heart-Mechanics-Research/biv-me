import os
import pandas as pd
import numpy as np
import statistics

from bivme.preprocessing.dicom.src.viewselection import ViewSelector
from bivme.preprocessing.dicom.src.predict_views import predict_views
from bivme.preprocessing.dicom.src.utils import write_sliceinfofile

CONFIDENCE_THRESHOLD = 0.75  # Modify this to change the confidence threshold for view selection. If metadata and image-based predictions disagree, the image-based prediction will be used if its confidence is above this threshold. 
                            # Otherwise, the metadata-based prediction will be used.

def handle_duplicates(view_predictions, viewSelector, my_logger):
    ## Remove duplicates
    # Type 1 - Same location, different series
    slice_locations = [] # Only consider slices not already excluded
    idx = []
    # Loop over view predictions
    for i, row in view_predictions.iterrows():
        if row['Predicted View'] == 'Excluded':
            continue
        slice_locations.append(viewSelector.df[viewSelector.df['Series Number'] == row['Series Number']]['Image Position Patient'].values[0])
        # Index should be the same as the row index in viewSelector.df
        index = viewSelector.df[viewSelector.df['Series Number'] == row['Series Number']].index[0]
        idx.append(index)

    repeated_slice_locations = [x for x in slice_locations if slice_locations.count(x) > 1]
    idx = [index for i,index in enumerate(idx) if slice_locations[i] in repeated_slice_locations]

    # Find repeated slice locations
    if len(idx) == 0:
        # my_logger.info('No duplicate slice locations found.')
        pass
    else:
        repeated_series = viewSelector.df.iloc[idx]
        repeated_series_num = repeated_series['Series Number'].values
        # Order by series number, so that if that if two series have the same Confidence, the higher series number is retained
        repeated_series_num = sorted(repeated_series_num, reverse=True)
        repeated_series_num = np.array(repeated_series_num)

        # Retain only the series with the highest Confidence, convert the rest to 'Excluded'
        confidences = [view_predictions[view_predictions['Series Number'] == x]['Confidence'].values[0] for x in repeated_series_num]

        idx_max = np.argmax(confidences)
        idx_to_exclude = [i for i in range(len(repeated_series_num)) if i != idx_max]

        view_predictions.loc[view_predictions['Series Number'].isin(repeated_series_num[idx_to_exclude]), 'Predicted View'] = 'Excluded'

        my_logger.info(f'Excluded series {repeated_series_num[idx_to_exclude]} as they exist at the same slice location as another series.') # TODO: Log which series

    # Type 2 - Multiple series classed as the same 'exclusive' view (i.e. 2ch, 3ch, 4ch, RVOT, RVOT-T, 2ch-RT, LVOT) 
    # i.e. a view that should only have one series 
    exclusive_views = ['2ch', '3ch', '4ch', 'RVOT', 'RVOT-T', '2ch-RT', 'LVOT']
    for view in exclusive_views:
        series = view_predictions[view_predictions['Predicted View'] == view]
        series_nums = series['Series Number'].values
        # Order by series number, so that if that if two series have the same Confidence, the higher series number is retained
        series_nums = sorted(series_nums, reverse=True)
        series_nums = np.array(series_nums)

        if len(series) > 1:
            my_logger.info(f'Multiple series classed as {view} ({series_nums}).')

            confidences = [view_predictions[view_predictions['Series Number'] == x]['Confidence'].values[0] for x in series_nums]

            idx_max = np.argmax(confidences)
            idx_to_exclude = [i for i in range(len(series)) if i != idx_max]
            view_predictions.loc[view_predictions['Series Number'].isin(series_nums[idx_to_exclude]), 'Predicted View'] = 'Excluded'

            my_logger.info(f'Excluded series {series_nums[idx_to_exclude]}')

    return view_predictions

def select_views(patient, src, dst, model, states, option, my_logger):
    if option == 'default':
        # Metadata-based model
        metadata_csv_path = os.path.join(dst, 'view-classification', 'metadata_view_predictions.csv')
        my_logger.info('Performing metadata-based view prediction...')
        viewSelectorMetadata = ViewSelector(src, dst, model, type='metadata', csv_path=metadata_csv_path, my_logger=my_logger)
        predict_views(viewSelectorMetadata)
        my_logger.success('Metadata-based view prediction complete.')

        # Image-based model
        my_logger.info('Performing image-based view prediction...')
        image_csv_path = os.path.join(dst, 'view-classification', 'image_view_predictions.csv')
        viewSelectorImage = ViewSelector(src, dst, model, type='image', csv_path=image_csv_path, my_logger=my_logger)
        predict_views(viewSelectorImage)
        my_logger.success('Image-based view prediction complete.')

        # Combine metadata and image-based predictions
        my_logger.info('Combining metadata and image-based view predictions...')
        metadata_view_predictions = pd.read_csv(metadata_csv_path)
        image_view_predictions = pd.read_csv(image_csv_path)

        all_series = list(set(np.concatenate([metadata_view_predictions['Series Number'].values,image_view_predictions['Series Number'].values])))
        view_predictions_array = []
        refinement_map = {'2ch': 'LAX', '2ch-RT': 'LAX', '3ch': 'LAX', '4ch': 'LAX', 'LVOT': 'SAX', 'OTHER': 'SAX', 'RVOT': 'Outflow', 'RVOT-T': 'Outflow', 'SAX': 'SAX', 'SAX-atria': 'SAX'}   # Map refined views to view types (SAX, LAX, Outflow)
                                                                                                                                                                                                # LVOT is considered SAX for the purposes of this model
        # Loop over all series and get the predictions from both metadata and image-based models                                                                                               
        for series in all_series:
            metadata_row = metadata_view_predictions[metadata_view_predictions['Series Number'] == series]
            image_row = image_view_predictions[image_view_predictions['Series Number'] == series]

            metadata_pred = metadata_row['Predicted View'].values[0]
            metadata_pred_type = refinement_map[metadata_pred]
            image_pred = image_row['Predicted View'].values[0]
            image_pred_type = refinement_map[image_pred]

            conf = image_row[f'{image_row["Predicted View"].values[0]} confidence'].values[0]
            vote_share = image_row['Vote Share'].values[0]

            # Scenario 1: Metadata and image-based predictions agree on view type. We trust the image-based prediction.
            if metadata_pred_type == image_pred_type:
                if conf < CONFIDENCE_THRESHOLD:
                    my_logger.warning(f"Low confidence for series {series} with image based prediction ({image_pred}). Metadata-based prediction is {metadata_pred}.") # TODO: Remove after debugging
                view_predictions_array.append([series, image_pred, vote_share, conf, image_row['Frames Per Slice'].values[0]])

            # Scenario 2: Metadata and image-based predictions disagree on view type. We use the metadata-based prediction to correct the image-based prediction
            elif (metadata_pred_type == 'SAX' and image_pred_type == 'LAX') or (metadata_pred_type == 'LAX' and image_pred_type == 'SAX'): 
                my_logger.warning(f'Series {series} metadata and image-based predictions conflict: {metadata_pred} ({metadata_pred_type}) vs {image_pred} ({image_pred_type}). Using metadata-based prediction to correct image-based prediction...') # TODO: Remove after debugging
                # Zero out the confidence of the incorrect categories
                confidences = []
                for v in list(refinement_map.keys()):
                    if refinement_map[v] != metadata_pred_type:
                        image_row[f'{v} confidence'] = 0
                    confidences.append(image_row[f'{v} confidence'].values[0])
                
                # Image prediction is the view with the highest confidence remaining
                image_pred = list(refinement_map.keys())[np.argmax(confidences)]
                conf = image_row[f'{image_pred} confidence'].values[0]
                vote_share = 0
                view_predictions_array.append([series, image_pred, vote_share, conf, image_row['Frames Per Slice'].values[0]])

            # Scenario 3: Metadata and image-based predictions disagree on view type, and the image-based prediction has low confidence. We use the metadata-based prediction.
            elif conf < CONFIDENCE_THRESHOLD:
                if metadata_pred_type == 'SAX': # Metadata model performs poorly distinguishing between SAX type views, so we use to it correct the image-based model instead
                    my_logger.warning(f'Low confidence for series {series} with image based prediction: {image_pred} ({image_pred_type}). Using metadata-based prediction to correct to a SAX type view...') # TODO: Remove after debugging
                    # Zero out the confidence of the incorrect categories
                    confidences = []
                    for v in list(refinement_map.keys()):
                        if refinement_map[v] != metadata_pred_type:
                            image_row[f'{v} confidence'] = 0
                        confidences.append(image_row[f'{v} confidence'].values[0])
                    
                    # Image prediction is the view with the highest confidence remaining
                    image_pred = list(refinement_map.keys())[np.argmax(confidences)]
                    conf = image_row[f'{image_pred} confidence'].values[0]
                    vote_share = 0

                    view_predictions_array.append([series, image_pred, vote_share, conf, image_row['Frames Per Slice'].values[0]])

                else: # Otherwise just use the metadata-based prediction directly
                    my_logger.warning(f'Low confidence for series {series} with image based prediction: {image_pred} ({image_pred_type}). Using metadata-based prediction ({metadata_pred}) instead...') # TODO: Remove after debugging
                    conf = 0 # Set confidence and vote share to 0 as we are not *that* confident in the metadata-based prediction
                    vote_share = 0
                    view_predictions_array.append([series, metadata_pred, vote_share, conf, image_row['Frames Per Slice'].values[0]])

            # Scenario 4: Metadata and image-based predictions disagree on view type, but the image-based prediction has high confidence. We trust the image-based prediction.
            elif conf >= CONFIDENCE_THRESHOLD:
                my_logger.warning(f'Series {series} metadata and image-based predictions conflict: {metadata_pred} ({metadata_pred_type}) vs {image_pred} ({image_pred_type}). Using image-based prediction for now...') # TODO: Remove after debugging
                view_predictions_array.append([series, image_pred, vote_share, conf, image_row['Frames Per Slice'].values[0]])
                
        view_predictions = pd.DataFrame(view_predictions_array, columns=['Series Number', 'Predicted View', 'Vote Share', 'Confidence', 'Frames Per Slice'])
        csv_path = os.path.join(dst, 'view-classification', 'view_predictions.csv')

        os.remove(metadata_csv_path)
        os.remove(image_csv_path)
        
        # Rename for simplicity
        viewSelector = viewSelectorImage
        viewSelector.csv_path = csv_path

        ## Flag any slices with non-matching number of phases
        # Use the SAX series as the reference for the 'right' number of phases
        try:
            sax_series = view_predictions[view_predictions['Predicted View'] == 'SAX'] 
            num_phases = statistics.mode(sax_series['Frames Per Slice'].values)
        except statistics.StatisticsError: # If no mode found (i.e. two values with equally similar counts), use median
            num_phases = np.median(sax_series['Frames Per Slice'].values)
        
        for i, row in viewSelector.df.iterrows():
            if row['Frames Per Slice'] != num_phases:
                my_logger.warning(f"Series {row['Series Number']} has a mismatching number of phases ({row['Frames Per Slice']} vs {num_phases}).")

        view_predictions = handle_duplicates(view_predictions, viewSelector, my_logger) # If duplicate slices are found, choose which ones to keep based on quality (approximated by confidence in prediction)

        # Print summary to log
        my_logger.success(f'View predictions for {patient}:')
        for view in view_predictions['Predicted View'].unique():
            my_logger.info(f'{view}: {len(view_predictions[view_predictions["Predicted View"] == view])} series')

        # Sort by series number
        view_predictions = view_predictions.sort_values('Series Number')

        # Write view predictions to csv
        view_predictions.to_csv(csv_path, mode='w', index=False)

        # Save to states folder
        states_path = os.path.join(states, 'view_predictions.csv')
        view_predictions.to_csv(states_path, mode='w', index=False)

        # Write pngs into respective view folders
        viewSelector.write_sorted_pngs()

    elif option == 'image-only':
        # Image-based model
        my_logger.info('Performing image-based view prediction...')
        csv_path = os.path.join(dst, 'view-classification', 'view_predictions.csv')
        viewSelector = ViewSelector(src, dst, model, type='image', csv_path=csv_path, my_logger=my_logger)
        predict_views(viewSelector)
        my_logger.success('Image-based view prediction complete.')

        view_predictions = pd.read_csv(csv_path)
        
        # Restucture dataframe
        all_series = view_predictions['Series Number'].values
        view_predictions_array = []                                                                               
        for series in all_series:
            image_row = view_predictions[view_predictions['Series Number'] == series]
            image_pred = image_row['Predicted View'].values[0]

            conf = image_row[f'{image_row["Predicted View"].values[0]} confidence'].values[0]
            vote_share = image_row['Vote Share'].values[0]

            view_predictions_array.append([series, image_pred, vote_share, conf, image_row['Frames Per Slice'].values[0]])
        view_predictions = pd.DataFrame(view_predictions_array, columns=['Series Number', 'Predicted View', 'Vote Share', 'Confidence', 'Frames Per Slice'])

        ## Flag any slices with non-matching number of phases
        # Use the SAX series as the reference for the 'right' number of phases
        try:
            sax_series = view_predictions[view_predictions['Predicted View'] == 'SAX'] 
            num_phases = statistics.mode(sax_series['Frames Per Slice'].values)
        except statistics.StatisticsError: # If no mode found (i.e. two values with equally similar counts), use median
            num_phases = np.median(sax_series['Frames Per Slice'].values)
        
        for i, row in viewSelector.df.iterrows():
            if row['Frames Per Slice'] != num_phases:
                my_logger.warning(f"Series {row['Series Number']} has a mismatching number of phases ({row['Frames Per Slice']} vs {num_phases}).")

        view_predictions = handle_duplicates(view_predictions, viewSelector, my_logger) # If duplicate slices are found, choose which ones to keep based on quality (approximated by confidence in prediction)

        # Print summary to log
        my_logger.success(f'View predictions for {patient}:')
        for view in view_predictions['Predicted View'].unique():
            my_logger.info(f'{view}: {len(view_predictions[view_predictions["Predicted View"] == view])} series')

        # Sort by series number
        view_predictions = view_predictions.sort_values('Series Number')

        # Write view predictions to csv
        view_predictions.to_csv(csv_path, mode='w', index=False)

        # Save to states folder
        states_path = os.path.join(states, 'view_predictions.csv')
        view_predictions.to_csv(states_path, mode='w', index=False)

        # Write pngs into respective view folders
        viewSelector.write_sorted_pngs()

    
    elif option == 'load':
        my_logger.info('Loading view predictions from states folder...')

        csv_path = os.path.join(states, 'view_predictions.csv')
        if not os.path.exists(csv_path):
            my_logger.error(f'View predictions not found at {csv_path}. Please run view selection with option="default" or "image-only" first.')
            raise FileNotFoundError(f'View predictions not found at {csv_path}. Please run view selection with option="default" or "image-only" first.')
        
        view_predictions = pd.read_csv(csv_path)

        viewSelector = ViewSelector(src, dst, model, type='image', csv_path=csv_path, my_logger=my_logger)
        viewSelector.load_predictions()

        ## Flag any slices with non-matching number of phases
        # Use the SAX series as the reference for the 'right' number of phases
        try:
            sax_series = view_predictions[view_predictions['Predicted View'] == 'SAX'] 
            num_phases = statistics.mode(sax_series['Frames Per Slice'].values)
        except statistics.StatisticsError: # If no mode found (i.e. two values with equally similar counts), use median
            num_phases = np.median(sax_series['Frames Per Slice'].values)
        
        for i, row in viewSelector.df.iterrows():
            if row['Frames Per Slice'] != num_phases:
                my_logger.warning(f"Series {row['Series Number']} has a mismatching number of phases ({row['Frames Per Slice']} vs {num_phases}).")

        # Print summary
        my_logger.success(f'View predictions for {patient}:')
        for view in view_predictions['Predicted View'].unique():
            my_logger.info(f'{view}: {len(view_predictions[view_predictions["Predicted View"] == view])} series')

        # Write csv to dst
        view_predictions.to_csv(os.path.join(dst, 'view-classification', 'view_predictions.csv'), mode='w', index=False)

    else:
        raise ValueError('Invalid option. Please use "default", "image-only", or "load".')

    out = []
    for i, row in view_predictions.iterrows():
        # Get row of viewSelector.df
        series_row = viewSelector.df[viewSelector.df['Series Number'] == row['Series Number']].iloc[0]
        frames_per_slice = series_row['Frames Per Slice']
        out.append([series_row['Series Number'], frames_per_slice, series_row['Filename'], row['Predicted View'], series_row['Image Position Patient'], series_row['Image Orientation Patient'], series_row['Pixel Spacing'], series_row['Img']])

    # generate dataframe
    slice_info_df = pd.DataFrame(out, columns = ['Slice ID', 'Frames Per Slice', 'File', 'View', 'ImagePositionPatient', 'ImageOrientationPatient', 'Pixel Spacing', 'Img'])

    # write slice info file
    slice_mapping = write_sliceinfofile(dst, slice_info_df)
    
    return slice_info_df, num_phases, slice_mapping
