import os
import shutil

from bivme.preprocessing.dicom.src.sliceviewer import SliceViewer

def export_guidepoints(dst, output, slice_dict, slice_mapping):
    # check if files in output folder already exist
    if os.path.exists(output):
        existing_files = os.listdir(output)
        for file in existing_files:
            if file.endswith('.txt'):
                os.remove(os.path.join(output, file))
            
    for s in slice_dict.values():
        s.export_slice(output, slice_mapping)

    # Move slice info file to output folder
    shutil.copyfile(os.path.join(dst, 'SliceInfoFile.txt'), os.path.join(output, 'SliceInfoFile.txt'))
    