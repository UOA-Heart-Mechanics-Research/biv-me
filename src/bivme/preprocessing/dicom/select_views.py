import os,sys
import pandas as pd
import numpy as np
import statistics
import warnings
warnings.filterwarnings('ignore')

from bivme.preprocessing.dicom.src.viewselection import ViewSelector
from bivme.preprocessing.dicom.src.predict_views import predict_views
from bivme.preprocessing.dicom.src.utils import write_sliceinfofile

def select_views(patient, src, dst, model, states, option, my_logger):
    if option == 'default':
        csv_path = os.path.join(dst, 'view-classification', 'view_predictions.csv')
        viewSelector = ViewSelector(src, dst, model, csv_path=csv_path, my_logger=my_logger)
        predict_views(viewSelector)

        view_predictions = pd.read_csv(csv_path)

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
            my_logger.info('No duplicate slice locations found.')
        else:
            repeated_series = viewSelector.df.iloc[idx]
            repeated_series_num = repeated_series['Series Number'].values
            # Order by series number, so that if that if two series have the same confidence, the higher series number is retained
            repeated_series_num = sorted(repeated_series_num, reverse=True)
            repeated_series_num = np.array(repeated_series_num)

            # Retain only the series with the highest confidence, convert the rest to 'Excluded'
            confidences = [view_predictions[view_predictions['Series Number'] == x]['Confidence'].values[0] for x in repeated_series_num]

            idx_max = np.argmax(confidences)
            idx_to_exclude = [i for i in range(len(repeated_series_num)) if i != idx_max]

            view_predictions.loc[view_predictions['Series Number'].isin(repeated_series_num[idx_to_exclude]), 'Predicted View'] = 'Excluded'

            my_logger.info(f'Excluded series {repeated_series_num[idx_to_exclude]} due to duplicate slice location.')
    
        # Type 2 - Multiple series classed as the same 'exclusive' view (i.e. 2ch, 3ch, 4ch, RVOT, RVOT-T 2ch-RT, RVOT-T, LVOT) 
        # i.e. a view that should only have one series 
        exclusive_views = ['2ch', '3ch', '4ch', 'RVOT', 'RVOT-T', '2ch-RT', 'LVOT']
        for view in exclusive_views:
            series = view_predictions[view_predictions['Predicted View'] == view]
            series_nums = series['Series Number'].values
            # Order by series number, so that if that if two series have the same confidence, the higher series number is retained
            series_nums = sorted(series_nums, reverse=True)
            series_nums = np.array(series_nums)

            if len(series) > 1:
                my_logger.info(f'Multiple series classed as {view}.')

                confidences = [view_predictions[view_predictions['Series Number'] == x]['Confidence'].values[0] for x in series_nums]

                idx_max = np.argmax(confidences)
                idx_to_exclude = [i for i in range(len(series)) if i != idx_max]
                view_predictions.loc[view_predictions['Series Number'].isin(series_nums[idx_to_exclude]), 'Predicted View'] = 'Excluded'

                my_logger.info(f'Excluded series {series_nums[idx_to_exclude]} due to multiple series classed as {view}.')

        # Print summary to log
        my_logger.info(f'View predictions for {patient}:')
        for view in view_predictions['Predicted View'].unique():
            my_logger.info(f'{view}: {len(view_predictions[view_predictions["Predicted View"] == view])} series')

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
            my_logger.error(f'View predictions not found at {csv_path}. Please run view selection with option="default" first.')
            raise FileNotFoundError(f'View predictions not found at {csv_path}. Please run view selection with option="default" first.')
        
        view_predictions = pd.read_csv(csv_path)

        viewSelector = ViewSelector(src, dst, model, csv_path=csv_path, my_logger=my_logger)
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
        my_logger.info(f'View predictions for {patient}:')
        for view in view_predictions['Predicted View'].unique():
            my_logger.info(f'{view}: {len(view_predictions[view_predictions["Predicted View"] == view])} series')

        # Write csv to dst
        view_predictions.to_csv(os.path.join(dst, 'view-classification', 'view_predictions.csv'), mode='w', index=False)
        

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
