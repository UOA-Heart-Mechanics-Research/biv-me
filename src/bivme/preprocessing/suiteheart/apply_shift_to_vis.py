import os
import sys
from pathlib import Path
import pyvista as pv

sys.path.append(r"C:\Users\jdil469\Code\biv-me")
from bivme.fitting.GPDataSet import *


if __name__ == "__main__":
    '''
    This script is used to apply slice shift to contours stored as vtk files for the purpose of visualisation.
    First, it calculates the slice shift for each frame in the GPFile and then applies the shift to the contours in the vtk files 
    and rewrites them.

    Author: Joshua Dillon
    Last updated: 2024-06-19
    '''

    dir_gp = r"R:\resmed201900006-biomechanics-in-heart-disease\Sandboxes\Josh\projects\bivme\suiteheart\gpfiles\processed"
    dir_vtk = r"R:\resmed201900006-biomechanics-in-heart-disease\Sandboxes\Josh\collaborations\suiteheart\visualisation"
    dir_out = r"R:\resmed201900006-biomechanics-in-heart-disease\Sandboxes\Josh\collaborations\suiteheart\visualisation-corrected"

    caselist = ["cardiohance_022"]
    casedirs = [Path(dir_gp, case).as_posix() for case in caselist]

    for folder in casedirs:
        all_frames = pd.read_csv(os.path.join(folder, "GPFile_clean.txt"), sep="\t")
        frames_to_fit = sorted(np.unique([i[6] for i in all_frames.values]))

        data_set = []
        for frame in frames_to_fit:
            data_set.append(
                GPDataSet(
                os.path.join(folder, "GPFile_clean.txt"),
                os.path.join(folder, "SliceInfoFile.txt"),
                os.path.basename(folder),
                sampling=1,
                time_frame_number=frame,
            )
            )
        
        offsets = []
        for data in data_set:
            offset = data.sinclaire_slice_shifting(frame_num=data.time_frame)
            offsets.append(offset)

        print('Offsets calculated. Applying to vtk files...')
        # Read in vtks
        vtk_case = os.path.join(dir_vtk, os.path.basename(folder))
        vtk_out = os.path.join(dir_out, os.path.basename(folder))
        if not os.path.exists(vtk_out):
            os.makedirs(vtk_out)

        slice_numbers = np.unique(data.slice_number)

        for frame in frames_to_fit:
            offset = offsets[frame-1]
            if frame < 10:
                frame_str = f'00{frame}'
            else:
                frame_str = f'0{frame}'

            vtk_files = [f for f in os.listdir(vtk_case) if f.endswith(f'{frame_str}.vtk')]
            for vtk_file in vtk_files:
                # Read in vtk
                mesh = pv.read(os.path.join(vtk_case, vtk_file))
                # Get img2world transform
                slice_num = int(vtk_file.split('_')[3].replace('lax','').replace('sax',''))
                if 'lax' in vtk_file:
                    slice_num = slice_num + max(slice_numbers) - 3 # always 3 lax slices
                idx = np.where(slice_num==slice_numbers)[0][0]
                translation2d = offset[0][idx]
                img2world = data.frames[slice_num].get_affine_matrix()
                # Convert 2D shift to 3D shift
                translation2d = np.array([translation2d[0], translation2d[1], 0, 1])
                translation3d = np.dot(translation2d, img2world.T)
                translation3d = translation3d[:3] - offset[1][idx]
                # Apply shift
                points = mesh.points
                points[:,0] += translation3d[0]
                points[:,1] += translation3d[1]
                points[:,2] += translation3d[2]
                mesh.points = points
                mesh.save(os.path.join(vtk_out, vtk_file))
                

            

