import os,sys
import pandas as pd
import numpy as np
from pathlib import Path

# add bivme to path
sys.path.append(r"C:\Users\jdil469\Code\biv-me")

from bivme.preprocessing.utils import Clean_contours
from bivme.fitting.GPDataSet import GPDataSet

if __name__ == "__main__":
    '''
    Cleans LAX contours from GPFile

    Author: Joshua Dillon
    Last updated: 2024-06-19
    '''


    # set directory containing GPFile and SliceInfoFile
    dir_gp = r"R:\resmed201900006-biomechanics-in-heart-disease\Sandboxes\Debbie\collaborations\stf\bivme\processed"

    # caselist = ["cardiohance_022"]
    caselist = os.listdir(dir_gp)
    casedirs = [Path(dir_gp, case).as_posix() for case in caselist]

    initial_gpfile = "GPFile.txt"
    initial_sliceinfo = "SliceInfoFile.txt"

    for folder in casedirs:
        print(f"Cleaning {folder}")
        all_frames = pd.read_csv(os.path.join(folder, initial_gpfile), sep="\t")
        frames_to_fit = sorted(np.unique([i[6] for i in all_frames.values]))
        data_set = []
        for num in frames_to_fit:
            data_set.append(
                (
                    num,
                    GPDataSet(
                        os.path.join(folder, initial_gpfile),
                        os.path.join(folder, initial_sliceinfo),
                        os.path.basename(folder),
                        sampling=1,
                        time_frame_number=num,
                    ),
                )
            )
        #TODO: Clean contours from 3Ch slice
        for i in range(0,len(data_set)):
            data_set[i][1].clean_MV_3D()
        Clean_contours(folder, data_set, "GPFile_clean-test.txt")