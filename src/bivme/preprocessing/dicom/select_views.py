import os
import pandas as pd
import numpy as np
import statistics

from bivme.preprocessing.dicom.src.viewselection import ViewSelector
from bivme.preprocessing.dicom.src.predict_views import predict_views
from bivme.preprocessing.dicom.src.viewcorrection import VSGUI


CONFIDENCE_THRESHOLD = 0.66 # Modify this to change the confidence threshold for view selection. If metadata and image-based predictions disagree, the image-based prediction will be used if its confidence is above this threshold. 
                            # Otherwise, the metadata-based prediction will be used.

def handle_duplicates(view_predictions, viewSelector, my_logger):
    excluded_series = []

    ## Remove duplicates
    # Type 1 - Same location, different series
    slice_locations = [] # Only consider slices not already excluded
    indices = []
    # Loop over view predictions
    for i, row in view_predictions.iterrows():
        if row['Predicted View'] == 'Excluded':
            continue
        slice_locations.append(viewSelector.df[viewSelector.df['Series Number'] == row['Series Number']]['Image Position Patient'].values[0])
        # Index should be the same as the row index in viewSelector.df
        index = viewSelector.df[viewSelector.df['Series Number'] == row['Series Number']].index[0]
        indices.append(index)

    repeated_slice_locations = [x for x in slice_locations if slice_locations.count(x) > 1]
    repeated_slice_locations = list(set(repeated_slice_locations))  # Convert to unique set

    for repeated_slice_location in repeated_slice_locations:
        idx = [index for i,index in enumerate(indices) if slice_locations[i] == repeated_slice_location]

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

            for i in idx_to_exclude:
                original_vp = view_predictions[view_predictions['Series Number'] == repeated_series_num[i]]['Predicted View'].values[0]

                # Format is series number, original view prediction, reason for exclusion, series number of kept series
                excluded_series.append([repeated_series_num[i], original_vp, 'Same location', repeated_series_num[idx_max]])
                my_logger.info(f'Excluded series {repeated_series_num[i]} as it exists at the same slice location as another series ({repeated_series_num[idx_max]}).')

                # Exclude from predictions
                view_predictions.loc[view_predictions['Series Number'] == repeated_series_num[i], 'Predicted View'] = 'Excluded'

    # Type 2 - Multiple series classed as the same 'exclusive' view (i.e. 2ch, 3ch, 4ch, RVOT, RVOT-T, 2ch-RT, LVOT) 
    # i.e. a view that should only have one series 
    exclusive_views = ['2ch', '3ch', '4ch', 'RVOT', 'RVOT-oblique', '2ch-RV', 'LVOT']
    for view in exclusive_views:
        series = view_predictions[view_predictions['Predicted View'] == view]
        series_nums = series['Series Number'].values
        # Order by series number, so that if that if two series have the same Confidence, the higher series number is retained
        series_nums = sorted(series_nums, reverse=True)
        series_nums = np.array(series_nums)

        if len(series) > 1:
            confidences = [view_predictions[view_predictions['Series Number'] == x]['Confidence'].values[0] for x in series_nums]

            idx_max = np.argmax(confidences)
            idx_to_exclude = [i for i in range(len(series)) if i != idx_max]

            for i in idx_to_exclude:
                original_vp = view_predictions[view_predictions['Series Number'] == series_nums[i]]['Predicted View'].values[0]

                # Format is series number, original view prediction, reason for exclusion, series number of kept series
                excluded_series.append([series_nums[i], original_vp, 'Same exclusive view', series_nums[idx_max]])
                my_logger.info(f'Excluded series {series_nums[i]} as belongs to same exclusive view ({original_vp}) as {series_nums[idx_max]}.')

                # Exclude from predictions
                view_predictions.loc[view_predictions['Series Number'] == series_nums[i], 'Predicted View'] = 'Excluded'
            
    excluded_df = pd.DataFrame(excluded_series, columns=['Excluded Series', 'Original View', 'Reason', 'Kept Series'])

    return view_predictions, excluded_df


def correct_views(patient, dst, viewSelector, csv_path, states_path, show_videos, my_logger):
    gui = VSGUI(patient, dst, viewSelector, show_videos, my_logger)
    gui.correct_views_gui()

    # Load the corrected predictions
    view_predictions = pd.read_csv(csv_path)
    viewSelector.show_warnings = False
    viewSelector.load_predictions()
    view_predictions.to_csv(states_path, mode='w', index=False) # Save to states path

    my_logger.success('View correction complete. Predictions saved.')

    return view_predictions, viewSelector


def select_views(patient, src, dst, model, states, config, my_logger):
    # Unpack config
    option = config["view-selection"]["option"] # 'default', 'metadata-only', 'image-only', 'load'
    correct_mode = config["view-selection"]["correct_mode"] # 'manual', 'adaptive', 'automatic'
    show_videos = config["view-selection"]["show_videos"] # true or false

    if option == 'default':
        # Metadata-based model
        metadata_csv_path = os.path.join(dst, 'view-classification', 'metadata_view_predictions.csv')
        my_logger.info('Performing metadata-based view prediction...')
        viewSelectorMetadata = ViewSelector(src, dst, model, type='metadata', csv_path=metadata_csv_path, show_warnings=True, my_logger=my_logger)
        predict_views(viewSelectorMetadata)
        my_logger.success('Metadata-based view prediction complete.')

        # Image-based model
        my_logger.info('Performing image-based view prediction...')
        image_csv_path = os.path.join(dst, 'view-classification', 'image_view_predictions.csv')
        viewSelectorImage = ViewSelector(src, dst, model, type='image', csv_path=image_csv_path, show_warnings=False, my_logger=my_logger)
        predict_views(viewSelectorImage)
        my_logger.success('Image-based view prediction complete.')

        # Combine metadata and image-based predictions
        my_logger.info('Combining metadata and image-based view predictions...')
        metadata_view_predictions = pd.read_csv(metadata_csv_path)
        image_view_predictions = pd.read_csv(image_csv_path)

        all_series = list(set(np.concatenate([metadata_view_predictions['Series Number'].values,image_view_predictions['Series Number'].values])))
        view_predictions_array = []

        old_refinement_map = {'2ch': 'LAX', '2ch-RT': 'LAX', '3ch': 'LAX', '4ch': 'LAX', 'LVOT': 'LAX', 'OTHER': 'SAX', 'RVOT': 'LAX', 'RVOT-T': 'Other', 'SAX': 'SAX', 'SAX-atria': 'SAX'}   
        refinement_map = {'2ch': 'LAX', '2ch-RV': 'LAX', '3ch': 'LAX', '4ch': 'LAX', 'LVOT': 'LAX', 'SAX-other': 'SAX', 'RVOT': 'LAX', 'RVOT-oblique': 'Other', 'SAX': 'SAX', 'SAX-atria': 'SAX'}   # Map refined views to view types (SAX, LAX, Outflow)        

        # Loop over all series and get the predictions from both metadata and image-based models                                                                                               
        for series in all_series:
            metadata_row = metadata_view_predictions[metadata_view_predictions['Series Number'] == series]
            image_row = image_view_predictions[image_view_predictions['Series Number'] == series]

            metadata_pred_type = metadata_row['View Class'].values[0]

            image_pred = image_row['Predicted View'].values[0]
            image_pred_type = refinement_map[image_pred]

            conf = image_row[f'{image_row["Predicted View"].values[0]} confidence'].values[0]
            vote_share = image_row['Vote Share'].values[0]

            description = image_row['Series Description'].values[0]
            location = image_row['Slice Location'].values[0]

            # Scenario 1: Metadata and image-based predictions agree on view type. We trust the image-based prediction.
            if metadata_pred_type == image_pred_type:
                if conf < CONFIDENCE_THRESHOLD:
                    my_logger.warning(f"Low confidence for series {series} with image based prediction ({image_pred})...") 
                view_predictions_array.append([series, description, location, image_pred, vote_share, conf, image_row['Frames Per Slice'].values[0]])

            # Scenario 2: Metadata and image-based predictions disagree on view type. We use the metadata-based prediction to correct the image-based prediction
            elif (metadata_pred_type != image_pred_type):
                my_logger.warning(f'Series {series} metadata and image-based predictions conflict: {metadata_pred_type} ({metadata_pred_type}) vs {image_pred} ({image_pred_type}). Using metadata-based prediction to correct image-based prediction...')
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
                view_predictions_array.append([series, description, location, image_pred, vote_share, conf, image_row['Frames Per Slice'].values[0]])

        view_predictions = pd.DataFrame(view_predictions_array, columns=['Series Number', 'Series Description', 'Slice Location', 'Predicted View', 'Vote Share', 'Confidence', 'Frames Per Slice'])
        csv_path = os.path.join(dst, 'view-classification', 'view_predictions.csv')

        # os.remove(metadata_csv_path)
        # os.remove(image_csv_path)
        
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

        view_predictions, excluded_df = handle_duplicates(view_predictions, viewSelector, my_logger) # If duplicate slices are found, choose which ones to keep based on quality (approximated by confidence in prediction)

        # Add excluded series info to view selector
        viewSelector.excluded_df = excluded_df

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

        # Corrections?
        if correct_mode == 'manual':
            my_logger.info('Manual corrections mode enabled. Launching view correction GUI...')
            # Run the view correction GUI
            view_predictions, viewSelector = correct_views(patient, dst, viewSelector, csv_path, states_path, show_videos, my_logger)

        elif correct_mode == 'adaptive':
            my_logger.info('Adaptive corrections mode enabled. Checking for missing/bad predictions...')
            
            # Check if there are missing views
            all_views = view_predictions['Predicted View'].unique()
            views_missing = []
            if not '2ch' in all_views:
                views_missing.append('2ch')
            if not '3ch' in all_views:
                views_missing.append('3ch')
            if not '4ch' in all_views:
                views_missing.append('4ch')
            if not 'SAX' in all_views:
                views_missing.append('SAX')
            
            if len(views_missing) > 0:
                my_logger.warning(f'Views {views_missing} are missing from the predictions.  Launching view correction GUI...')
                # Run the view correction GUI
                view_predictions, viewSelector = correct_views(patient, dst, viewSelector, csv_path, states_path, show_videos, my_logger)

            else:
                # Check for any predictions with low confidence 
                low_confidence = view_predictions[view_predictions['Confidence'] < CONFIDENCE_THRESHOLD]
                if len(low_confidence) > 0:
                    my_logger.warning(f'Found predictions with low confidence.  Launching view correction GUI...')
                    # Run the view correction GUI
                    view_predictions, viewSelector = correct_views(patient, dst, viewSelector, csv_path, states_path, show_videos, my_logger)
                else:
                    my_logger.success('No missing views or low confidence predictions found. No corrections needed (hopefully).')

        elif correct_mode == 'automatic':
            pass
        else:
            raise ValueError('Invalid correction mode. Please use "manual" or "automatic".')

    elif option == 'image-only':
        # Image-based model
        my_logger.info('Performing image-based view prediction...')
        csv_path = os.path.join(dst, 'view-classification', 'view_predictions.csv')
        viewSelector = ViewSelector(src, dst, model, type='image', csv_path=csv_path, show_warnings=True, my_logger=my_logger)
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

            description = image_row['Series Description'].values[0]
            location = image_row['Slice Location'].values[0]

            view_predictions_array.append([series, description, location, image_pred, vote_share, conf, image_row['Frames Per Slice'].values[0]])
            
        view_predictions = pd.DataFrame(view_predictions_array, columns=['Series Number', 'Series Description', 'Slice Location', 'Predicted View', 'Vote Share', 'Confidence', 'Frames Per Slice'])

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

        view_predictions, excluded_df = handle_duplicates(view_predictions, viewSelector, my_logger) # If duplicate slices are found, choose which ones to keep based on quality (approximated by confidence in prediction)

        # Add excluded series info to view selector
        viewSelector.excluded_df = excluded_df

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

        # Corrections?
        if correct_mode == 'manual':
            my_logger.info('Manual corrections mode enabled. Launching view correction GUI...')
            # Run the view correction GUI
            view_predictions, viewSelector = correct_views(patient, dst, viewSelector, csv_path, states_path, show_videos, my_logger)

        elif correct_mode == 'adaptive':
            my_logger.info('Adaptive corrections mode enabled. Checking for missing/bad predictions...')
            
            # Check if there are missing views
            all_views = view_predictions['Predicted View'].unique()
            views_missing = []
            if not '2ch' in all_views:
                views_missing.append('2ch')
            if not '3ch' in all_views:
                views_missing.append('3ch')
            if not '4ch' in all_views:
                views_missing.append('4ch')
            if not 'SAX' in all_views:
                views_missing.append('SAX')
            
            if len(views_missing) > 0:
                my_logger.warning(f'Views {views_missing} are missing from the predictions.  Launching view correction GUI...')
                # Run the view correction GUI
                view_predictions, viewSelector = correct_views(patient, dst, viewSelector, csv_path, states_path, show_videos, my_logger)

            else:
                # Check for any predictions with low confidence 
                low_confidence = view_predictions[view_predictions['Confidence'] < CONFIDENCE_THRESHOLD]
                if len(low_confidence) > 0:
                    my_logger.warning(f'Found predictions with low confidence.  Launching view correction GUI...')
                    # Run the view correction GUI
                    view_predictions, viewSelector = correct_views(patient, dst, viewSelector, csv_path, states_path, show_videos, my_logger)
                else:
                    my_logger.success('No missing views or low confidence predictions found. No corrections needed (hopefully).')

        elif correct_mode == 'automatic':
            pass
        else:
            raise ValueError('Invalid correction mode. Please use "manual", "adaptive", or "automatic".')
    
    elif option == 'load':
        my_logger.info('Loading view predictions from states folder...')

        csv_path = os.path.join(states, 'view_predictions.csv')
        if not os.path.exists(csv_path):
            my_logger.error(f'View predictions not found at {csv_path}. Please run view selection with option="default" or "image-only" first.')
            raise FileNotFoundError(f'View predictions not found at {csv_path}. Please run view selection with option="default" or "image-only" first.')
        
        view_predictions = pd.read_csv(csv_path)

        viewSelector = ViewSelector(src, dst, model, type='image', csv_path=csv_path, show_warnings=True, my_logger=my_logger)
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
        csv_path = os.path.join(dst, 'view-classification', 'view_predictions.csv')
        states_path = os.path.join(states, 'view_predictions.csv')
        view_predictions.to_csv(csv_path, mode='w', index=False)

        # Corrections?
        if correct_mode == 'manual':
            my_logger.info('Manual corrections mode enabled. Launching view correction GUI...')
            # Run the view correction GUI
            view_predictions, viewSelector = correct_views(patient, dst, viewSelector, csv_path, states_path, show_videos, my_logger)

        elif correct_mode == 'adaptive':
            my_logger.info('Adaptive corrections mode enabled. Checking for missing/bad predictions...')
            
            # Check if there are missing views
            all_views = view_predictions['Predicted View'].unique()
            views_missing = []
            if not '2ch' in all_views:
                views_missing.append('2ch')
            if not '3ch' in all_views:
                views_missing.append('3ch')
            if not '4ch' in all_views:
                views_missing.append('4ch')
            if not 'SAX' in all_views:
                views_missing.append('SAX')
            
            if len(views_missing) > 0:
                my_logger.warning(f'Views {views_missing} are missing from the predictions.  Launching view correction GUI...')
                # Run the view correction GUI
                view_predictions, viewSelector = correct_views(patient, dst, viewSelector, csv_path, states_path, show_videos, my_logger)

            else:
                # Check for any predictions with low confidence 
                low_confidence = view_predictions[view_predictions['Confidence'] < CONFIDENCE_THRESHOLD]
                if len(low_confidence) > 0:
                    my_logger.warning(f'Found predictions with low confidence.  Launching view correction GUI...')
                    # Run the view correction GUI
                    view_predictions, viewSelector = correct_views(patient, dst, viewSelector, csv_path, states_path, show_videos, my_logger)
                else:
                    my_logger.success('No missing views or low confidence predictions found. No corrections needed (hopefully).')

        elif correct_mode == 'automatic':
            pass
        else:
            raise ValueError('Invalid correction mode. Please use "manual", "adaptive", or "automatic".')

    else:
        raise ValueError('Invalid option. Please use "default", "metadata-only", "image-only", or "load".')
    
    # Recalculate after correection
    try:
        sax_series = view_predictions[view_predictions['Predicted View'] == 'SAX'] 
        num_phases = statistics.mode(sax_series['Frames Per Slice'].values)
    except statistics.StatisticsError: # If no mode found (i.e. two values with equally similar counts), use median
        num_phases = np.median(sax_series['Frames Per Slice'].values)

    out = []
    for i, row in view_predictions.iterrows():
        # Get row of viewSelector.df
        series_row = viewSelector.df[viewSelector.df['Series Number'] == row['Series Number']].iloc[0]
        frames_per_slice = series_row['Frames Per Slice']
        out.append([series_row['Series Number'], frames_per_slice, series_row['Filename'], row['Predicted View'], series_row['Image Position Patient'], series_row['Image Orientation Patient'], series_row['Pixel Spacing'], series_row['Img']])

    # generate dataframe
    slice_info_df = pd.DataFrame(out, columns = ['Slice ID', 'Frames Per Slice', 'File', 'View', 'ImagePositionPatient', 'ImageOrientationPatient', 'Pixel Spacing', 'Img'])
    
    return slice_info_df, num_phases
