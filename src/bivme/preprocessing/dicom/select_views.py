import os
import pandas as pd
import numpy as np
import statistics
import warnings
warnings.filterwarnings('ignore')


from bivme.preprocessing.dicom.src.viewselection import ViewSelector

def select_views(patient, src, dst, model, states, option='default'):
    if option == 'default':
        csv_path = os.path.join(dst, 'view-classification', 'view_predictions.csv')
        viewSelector = ViewSelector(src, dst, model, csv_path=csv_path)
        viewSelector.predict_views()

        view_predictions = pd.read_csv(csv_path)

        ## Exclude any slices with non-matching number of phases
        try:
            num_phases = statistics.mode(viewSelector.df['Frames Per Slice'].values)
        except statistics.StatisticsError: # If no mode found (i.e. two values with equally similar counts), use median
            num_phases = np.median(viewSelector.df['Frames Per Slice'].values)
        
        for i, row in viewSelector.df.iterrows():
            if row['Frames Per Slice'] != num_phases:
                view_predictions.loc[view_predictions['Series Number'] == row['Series Number'], 'Predicted View'] = 'Excluded'
                print(f"Excluded series {row['Series Number']} due to mismatched number of phases ({row['Frames Per Slice']} vs {num_phases}).")

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
            print('No duplicate slice locations found.')
        else:
            repeated_series = viewSelector.df.iloc[idx]
            repeated_series_num = repeated_series['Series Number'].values

            # Retain only the series with the highest confidence, convert the rest to 'Excluded'
            confidences = [view_predictions[view_predictions['Series Number'] == x]['Confidence'].values[0] for x in repeated_series_num]
            idx_max = np.argmax(confidences)
            idx_to_exclude = [i for i in range(len(repeated_series_num)) if i != idx_max]

            print(f'Excluded series {repeated_series_num[idx_to_exclude]} due to duplicate slice location.')
            view_predictions[view_predictions['Series Number'].isin(repeated_series_num[idx_to_exclude])]['Predicted View'] = 'Excluded' # TODO: Check if this is actually working
    

        # Type 2 - Multiple series classed as the same 'exclusive' view (i.e. 2ch, 3ch, 4ch, RVOT, RVOT-T 2ch-RT, RVOT-T, LVOT) 
        # i.e. a view that should only have one series 
        exclusive_views = ['2ch', '3ch', '4ch', 'RVOT', 'RVOT-T', '2ch-RT', 'LVOT']
        for view in exclusive_views:
            series = view_predictions[view_predictions['Predicted View'] == view]
            if len(series) > 1:
                print(f'Multiple series classed as {view}.')
                confidences = series['Confidence'].values
                idx_max = np.argmax(confidences)
                idx_to_exclude = [i for i in range(len(series)) if i != idx_max]
                print(f'Excluded series {series.iloc[idx_to_exclude]["Series Number"].values}')
                view_predictions.loc[series.index[idx_to_exclude], 'Predicted View'] = 'Excluded'

        # Print summary
        print(f'View predictions for {patient}:')
        for view in view_predictions['Predicted View'].unique():
            print(f'{view}: {len(view_predictions[view_predictions["Predicted View"] == view])} series')

        # Write view predictions to csv
        view_predictions.to_csv(csv_path, mode='w', index=False)

        # Save to states folder
        states_path = os.path.join(states, 'view_predictions.csv')
        view_predictions.to_csv(states_path, mode='w', index=False)

        # Write pngs into respective view folders
        viewSelector.write_sorted_pngs()
    
    elif option == 'load':
        print('Loading view predictions from states folder...')
        csv_path = os.path.join(states, 'view_predictions.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'View predictions not found at {csv_path}. Please run view selection with option="default" first.')
        
        view_predictions = pd.read_csv(csv_path)
        try:
            num_phases = statistics.mode(view_predictions['Frames Per Slice'].values)
        except:
            num_phases = np.median(view_predictions['Frames Per Slice'].values)

        viewSelector = ViewSelector(src, dst, model, csv_path=csv_path)
        viewSelector.load_predictions()

        # Print summary
        print(f'View predictions for {patient}:')
        for view in view_predictions['Predicted View'].unique():
            print(f'{view}: {len(view_predictions[view_predictions["Predicted View"] == view])} series')

        # Write csv to dst
        view_predictions.to_csv(os.path.join(dst, 'view-classification', 'view_predictions.csv'), mode='w', index=False)
        

    out = []
    for i, row in view_predictions.iterrows():
        # Get row of viewSelector.df
        series_row = viewSelector.df[viewSelector.df['Series Number'] == row['Series Number']].iloc[0]
        out.append([series_row['Series Number'], series_row['Filename'], row['Predicted View'], series_row['Image Position Patient'], series_row['Image Orientation Patient'], series_row['Pixel Spacing'], series_row['Img']])

    # generate dataframe
    slice_info_df = pd.DataFrame(out, columns = ['Slice ID', 'File', 'View', 'ImagePositionPatient', 'ImageOrientationPatient', 'Pixel Spacing', 'Img'])

    # Calculate a slice mapping (reformat to 1-numslices)
    slice_mapping = {}
    for i, row in slice_info_df.iterrows():
        slice_mapping[row['Slice ID']] = i+1
        
    # write to slice info file
    with open(os.path.join(dst, 'SliceInfoFile.txt'), 'w') as f:
        for i, row in slice_info_df.iterrows():
            sliceID = slice_mapping[row['Slice ID']]
            file = row['File']
            file = os.path.basename(file)
            view = row['View']
            imagePositionPatient = row['ImagePositionPatient']
            imageOrientationPatient = row['ImageOrientationPatient']
            pixelSpacing = row['Pixel Spacing']
            
            f.write('{}\t'.format(file))
            f.write('sliceID: \t')
            f.write('{}\t'.format(sliceID))
            f.write('ImagePositionPatient\t')
            f.write('{}\t{}\t{}\t'.format(imagePositionPatient[0], imagePositionPatient[1], imagePositionPatient[2]))
            f.write('ImageOrientationPatient\t')
            f.write('{}\t{}\t{}\t{}\t{}\t{}\t'.format(imageOrientationPatient[0], imageOrientationPatient[1], imageOrientationPatient[2],
                                                imageOrientationPatient[3], imageOrientationPatient[4], imageOrientationPatient[5]))
            f.write('PixelSpacing\t')
            f.write('{}\t{}\n'.format(pixelSpacing[0], pixelSpacing[1]))
    
    return slice_info_df, num_phases, slice_mapping
