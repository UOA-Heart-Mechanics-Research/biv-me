import os
import pydicom

# Update these if they don't work for your dataset
INCLUSION_TERMS = [''] # include only series that have any one of these terms in the description
EXCLUSION_TERMS = ['loc', 'molli', 't1', 't2', 'dense', 'scout', 'grid', 'flow', 'fl2d',
                   'single shot', 'report', 'document', 'segmentation', 'result', 'mapping', 'mag', 'psir', 'suiteheart',
                   'axial', 'coronal', 'transverse', 'cas', 'survey', 'nav', 'tpat', 't-pat', 'gad',
                   'cs_rt_10sl_jc', 'truefisp', 'catch', 'fl3d', 'exceptions', 'haste', 'oblique', 'perf',
                   'pc', 'phase', 'contrast', 'ao', 'ps:', 'pa cine ir', 'dynamic_test', 'bolus', 'gre', 'rr', '3d'] # exclude series with any one of these terms in the description

def extract_cines(src, dst, my_logger):
    # This function is used to preprocess the DICOM files before running the pipeline. 
    # It (hopefully) extracts only cine images and converts them to .dcm format. 

    # Create destination directory if it does not exist
    processed_dcm_dir = os.path.join(dst, 'processed-dicoms')
    os.makedirs(processed_dcm_dir, exist_ok=True) # cine .dcms will be saved here

    # Check the operating system
    opsystem = None
    if os.name == 'nt':  # Windows
        opsystem = 'windows'
    elif os.name == 'posix':  # Linux or macOS
        opsystem = 'posix'

    file_paths = []
    excluded_descriptions = {}
    total_images = 0
    # Get all files in the source directory
    for root, dirs, files in os.walk(src):
        if not files:  # Skip empty directories
            continue
        total_images += len(files)
        for file in files:
            # Read series description
            filepath = os.path.join(root, file)

            if len(filepath) > 260 and opsystem == 'windows':
                # Need to reduce filepath length for Windows
                filepath = '\\\\?\\' + filepath

            try:
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)  # Only read metadata, not pixel data for speed
            except:
                my_logger.warning(f'Could not read {file}. Might not be a DICOM file.')
                continue

            try:
                description = dcm.SeriesDescription.lower()  # lower case for easier comparison
            except:
                my_logger.warning(f'Could not find series description tag for {file}. Excluded for now.')
                continue

            # if description == '':
            #     my_logger.warning(f'Series description is empty for {file}. Excluded for now.')
            #     continue
            
            # Check if the description contains any of the inclusion terms and does not contain any of the exclusion terms
            if any(term in description for term in INCLUSION_TERMS) and not any(term in description for term in EXCLUSION_TERMS):
                file_paths.append(filepath)
            else:
                # Find offending term(s)
                offences = []
                for term in EXCLUSION_TERMS:
                    if term in description:
                        offences.append(term)
                        
                excluded_descriptions[description] = offences
    
    my_logger.info(f'Found {total_images} images in the source directory')
    my_logger.info(f'Extracted {len(file_paths)} which matched the inclusion criteria.')

    if len(excluded_descriptions) > 0:
        my_logger.warning(f'Excluded series descriptions:')
        for desc, offences in excluded_descriptions.items():
            if len(offences) > 0:
                my_logger.warning(f' - {desc} (excluded due to term(s): {", ".join(offences)})')
            else:
                my_logger.warning(f' - {desc} (no inclusion terms matched)')

        my_logger.info(f'If you think some images are being wrongly excluded, please update the INCLUSION_TERMS and EXCLUSION_TERMS in src/bivme/preprocessing/dicom/extract_cines.py')

    file_names = [os.path.basename(file) for file in file_paths]
    sets = list(set(file_names))  # Get unique file names to avoid processing duplicates

    for s in sets:
        # For each set, find all files with the same name
        files = [fp for fp in file_paths if os.path.basename(fp) == s]
        for file in files:
            f = os.path.basename(file)
            root = os.path.dirname(file)
           
            # Save the cine images to the destination directory as .dcm files
            if not f.endswith('.dcm'):
                f = f'{f}.dcm'  # Ensure the file has a .dcm extension

            dcm = pydicom.dcmread(file)
            
            if len(files) == 1: # If there's only one file with this name, just save it
                try:
                    dcm.save_as(os.path.join(processed_dcm_dir, f))
                except Exception as e:
                    my_logger.warning(f'Could not save {file} as {f}. Error: {e}')
                continue
            else:
                idx = files.index(file) # Get the index of the file in the set
                f = f.replace('.dcm', '')  # Remove .dcm extension for indexing (added back in next line)

                try:
                    dcm.save_as(os.path.join(processed_dcm_dir, f'{f}_{idx}.dcm'))
                except Exception as e:
                    my_logger.warning(f'Could not save {file} as {f}_{idx}.dcm. Error: {e}')