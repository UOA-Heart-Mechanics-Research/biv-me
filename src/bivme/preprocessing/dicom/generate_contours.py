from bivme.preprocessing.dicom.src.sliceviewer import SliceViewer

def generate_contours(case, case_dst, slice_info_df, num_phases, version, my_logger):
    slice_dict = {}
    views = ['SAX', '2ch', '3ch', '4ch', 'RVOT']
    
    for view in views:
        slice_rows = slice_info_df[slice_info_df['View'] == view]
        for index, row in slice_rows.iterrows():
            my_logger.info(f'Generating contours for {view} slice {row["Slice ID"]}...')

            slice_id = row['Slice ID']
            slice = SliceViewer(case, case_dst, slice_info_df, view, slice_id, num_phases//2, num_phases=num_phases, full_cycle=True, version = version)

            # Landmarks estimated from intersections of contours
            slice.get_initial_landmarks()
            slice_dict[slice_id] = slice

    return slice_dict
