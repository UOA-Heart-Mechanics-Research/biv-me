
import os 
import numpy as np
import csv
from pathlib import Path

from bivme.preprocessing.utils import *


def do_preprocessing(folder,initial_gpfile,initial_sliceinfo, **kwargs):

        '''
        Author: Laura Dal Toso
        Date: 20/10/2022
        --------------------------------------------------------------
        This is the main function, that does the pre-processing of GPFiles and SliceInfoFiles to make them 
        compatible with the BiVFitting scripts
        -------------------------------------------------------------

        Input: 
            - folder where the GPFile and SliceInfoFile are stored

        Output: 
            - processed GPFile and SliceInfoFile, ready for the fitting

        ''' 

        if 'iter_num' in kwargs:
            iter_num = kwargs.get('iter_num', None)
            pid = os.getpid()
            # assign a new process ID and a new CPU to the child process 
            # iter_num corresponds to the id number of the CPU where the process will be run
            os.system("taskset -cp %d %d" %(iter_num, pid))

        if 'id_Frame' in kwargs:
            # acquire .csv file containing patient_id, ES frame number, ED frame number if present
            case_frame_dict = kwargs.get('id_Frame', None)

        # First check of SliceInfo and GPFile structure:
        gpfile, sliceinfofile = ReformatFiles(folder, initial_gpfile, initial_sliceinfo)
        
        # chose which frames to upload from the GPFile
        all_frames = pd.read_csv(os.path.join(folder, gpfile), sep = '\t') 
        frames_to_fit = sorted(np.unique([i[6] for i in all_frames.values])) 
        case = os.path.basename(os.path.normpath(folder))

        if do_landmarks_tracking == True:

            print('landmark checking ..')

            data_set = tracking_landmarks(
                folder, gpfile , sliceinfofile, output_csv = landmarks_csv)
            if clean_contours ==True:
                print('Cleaning Contours ..')
                final_data_set = Clean_contours(folder, data_set,'GPFile_clean.txt')


        elif do_landmarks_tracking == False and clean_contours ==True:

            print('Cleaning Contours ..')
            filename = os.path.join(folder, gpfile)
            filenameInfo = os.path.join(folder, sliceinfofile)

            data_set = []
            for num in frames_to_fit:
                data_set.append((num, GPDataSet(filename,filenameInfo, case, 
                    sampling = 1, time_frame_number = num)))
            
            final_data_set = Clean_contours(folder, data_set,'GPFile_clean.txt')
            
        else:
            final_data_set = []
            for num in frames_to_fit:
                final_data_set.append((num, GPDataSet(os.path.join(folder, gpfile), 
                os.path.join(folder, sliceinfofile), case, sampling = 1, time_frame_number = num)))

        if find_EDES_frames == True:
            print('Finding ES frame ..')
            findED_ESframe(case, final_data_set)  


        DoneFile = Path(os.path.join(folder,'Done.txt'))
        DoneFile.touch(exist_ok = True)            
    
        return DoneFile

if __name__ == '__main__':

    cases_folder = './test'
    cases_list = [os.path.join(cases_folder, batch) for batch in os.listdir(
        cases_folder)][2:3]

    # do you want to check that all landmarks present at each frame?
    do_landmarks_tracking = False
    #do you need to delete some points from your intial data? 
    clean_contours = False
    # do you need to find which frames are ED and ES?
    find_EDES_frames = True

    initial_gpfile = 'GPFile.txt'
    initial_sliceinfo = 'SliceInfoFile.txt'

    if do_landmarks_tracking == True:
        fieldnames = ['patient', 'frames', 'MITRAL_VALVE', 'TRICUSPID_VALVE', 'AORTA_VALVE', 'APEX_POINT']
        landmarks_csv = './results/landmarks.csv' 
        
        with open(landmarks_csv, 'w') as f:
            writer = csv.DictWriter(f, fieldnames= fieldnames)
            writer.writeheader()

    if find_EDES_frames ==True:
        with open('./results/case_id_frame.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames= [ 'patient', 'ED', ' ES measured'])
            writer.writeheader()

    [do_preprocessing(folder,initial_gpfile,initial_sliceinfo) for folder in cases_list]


