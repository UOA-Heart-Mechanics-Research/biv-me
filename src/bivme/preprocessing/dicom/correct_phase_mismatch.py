from bivme.preprocessing.dicom.src.utils import resample_seg

def correct_phase_mismatch(dst, slice_info_df, num_phases, my_logger):
    # Often, LAX and SAX series do not have matching number of phases. In that case, we need to resample segmentations to have the same number of phases.
    # We will use the SAX series as the reference for the 'right' number of phases

    # Find any series with non-matching number of phases (num_phases is the number of phases in the SAX series)
    mismatch_dict = {}
    for i, row in slice_info_df.iterrows():
        if row['Frames Per Slice'] != num_phases:
            mismatch_dict[row['Slice ID']] = row['Frames Per Slice']
            my_logger.warning(f"Series {row['Slice ID']} has a mismatching number of phases ({row['Frames Per Slice']} vs {num_phases}).")
    
    if len(mismatch_dict) > 0:
        for series, num_frames in mismatch_dict.items():
            view = slice_info_df[slice_info_df['Slice ID'] == series]['View'].values[0]
            my_logger.info(f'Resampling segmentations for series {series} ({view}) to have {num_phases} phases...')
            resample_seg(dst, view, series, num_phases, my_logger)
        my_logger.success('Phase mismatch correction complete.')
    else:
        my_logger.success('No phase mismatches found. No resampling required.')