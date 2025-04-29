import os
import shutil

from bivme.preprocessing.dicom.src.sliceviewer import SliceViewer

def export_guidepoints(dst, output, slice_dict, slice_mapping):
    for s in slice_dict.values():
        s.export_slice(output, slice_mapping)

    # Move slice info file to output folder
    shutil.copyfile(os.path.join(dst, 'SliceInfoFile.txt'), os.path.join(output, 'SliceInfoFile.txt'))
    