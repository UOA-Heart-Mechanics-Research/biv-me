import os
import shutil

from bivme.preprocessing.dicom.src.sliceviewer import SliceViewer

def export_guidepoints(case, case_dst, output_folder, slice_dict, slice_mapping):
    output_folder = os.path.join(output_folder, case)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)


    for s in slice_dict.values():
        s.export_slice(output_folder, slice_mapping)

    # Move slice info file to output folder
    shutil.copyfile(os.path.join(case_dst, 'SliceInfoFile.txt'), os.path.join(output_folder, 'SliceInfoFile.txt'))
    