import os
import pydicom

# Update these if they don't work for your dataset
# INCLUSION_TERMS = ['cine', 'tf2d1', 'retro'] # include only series that have any one of these terms in the description
EXCLUSION_TERMS = ['loc', 'localizer', 'molli', 't1', 't2', 'dense', 'scout', 'grid', 'flow', 'fl2d', 'trufi'] # exclude series with any one of these terms in the description

def extract_cines(src, dst, my_logger):
    # This function is used to preprocess the DICOM files before running the pipeline. 
    # It (hopefully) extracts only cine images and converts them to .dcm format. 

    # Create destination directory if it does not exist
    processed_dcm_dir = os.path.join(dst, 'processed-dicoms')
    os.makedirs(processed_dcm_dir, exist_ok=True) # cine .dcms will be saved here

    # Get all files in the source directory
    for root, dirs, files in os.walk(src):
        for file in files:
            try:
                dcm = pydicom.dcmread(os.path.join(root, file))
            except:
                my_logger.warning(f'Could not read {file}. Might not be a DICOM file.')
                continue

            description = dcm.SeriesDescription.lower() # lower case for easier comparison

            if not any(term in description for term in EXCLUSION_TERMS):
                # Save the cine images to the destination directory as .dcm files
                if file.endswith('.dcm'):
                    dcm.save_as(os.path.join(processed_dcm_dir, file))
                else:
                    dcm.save_as(os.path.join(processed_dcm_dir, f'{file}.dcm'))