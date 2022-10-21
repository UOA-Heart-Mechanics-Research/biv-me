from __future__ import print_function
from perform_fit import *

import os
import numpy as np
import time
import concurrent.futures
from pathlib import Path
import multiprocessing, traceback, time
import psutil


class Process(multiprocessing.Process):

    def __init__(self, *args, **kwargs):
        multiprocessing.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        try:
            multiprocessing.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            #raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception

def split_in_chunks(a, n):
    '''
    This function splits the list a in n chunks.
    Otput: list made of n lists

    '''
    k, m = divmod(len(a), n)
    return list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
  


def split_and_run(cases_list, workers, **kwargs):
    '''
    This function splits the total workload in chunks of equal size and runs the code in parallel.
    Input:
    cases_folder = list of paths to GPFiles
    workers = number of CPUs to be used
    id_frame (optional): dictionaty with patient number and ED, ES frame number
    '''
    case_frame_dict = None

    if 'id_Frame' in kwargs:
        # acquire .csv file containing patient_id, ES frame number, ED frame number if present
        case_frame_dict = kwargs.get('id_Frame', None)

    # split the total workload in chunks of equal size
    n_chunks = int(np.ceil(len(cases_list)/workers))

    print('TOT CASES:', len(cases_list))
    print('TOT CPUs selected: ', workers )
    print('-----> DATA WILL BE SPLIT INTO ', n_chunks, ' chunks')
 

    BatchLog = Path(os.path.join('./results/' ,'BatchLog.txt'))
    BatchLog.touch(exist_ok=True)   

    if workers == 1:
        if case_frame_dict is not None:
            results = [perform_fitting (folder, id_Frame = case_frame_dict) for folder in cases_list]
        else:
            results = [perform_fitting (folder) for folder in cases_list]

    if n_chunks <=1:
        # spawn a number of child processes equal to the number of patients in each chunk
        with concurrent.futures.ProcessPoolExecutor(max_workers= workers) as executor:
            #the CPU affinity is changed by perform_fitting, so that each child process is assigned to one CPU 
            if case_frame_dict is not None:    
                future = [executor.submit(
                    perform_fitting, folder, iter_num = i,id_Frame = case_frame_dict) for i,folder in enumerate(cases_list)]
            else: 
                future = [executor.submit(
                    perform_fitting, folder, iter_num = i) for i,folder in enumerate(cases_list)]
 
            for i,f in enumerate(future):    
                try:
                    result = f.result()
                except Exception as e:
                    with open(BatchLog, 'a') as f:
                        f.write('\n \t Failed for case: '+ cases_list[i])

                    print('Exception for case:', cases_list[i])
                else:
                    print('result = ', result)            

    if n_chunks >1:
        # split data in n chunks:
        split_folders = split_in_chunks(cases_list, n_chunks)
        
        with open(BatchLog, 'w') as f:
            f.write('Division in chunks: ')

            for i,patients in enumerate(split_folders):
                f.write('\n Chunk number: '+ str(i+1)) 
                f.write('\t '+ str(patients)) 


        for subfolders in split_folders:
            
            # spawn a number of child processes equal to the number of patients in each chunk
            with concurrent.futures.ProcessPoolExecutor(max_workers= workers) as executor:
        
                    if case_frame_dict is not None:    
                        future = [executor.submit(
                            perform_fitting, folder, iter_num = i,id_Frame = case_frame_dict) for i,folder in enumerate(subfolders)]
                    else: 
                        future = [executor.submit(
                            perform_fitting, folder, iter_num = i) for i,folder in enumerate(subfolders)]


                    for i,f in enumerate(future):    
                        try:
                            result = f.result()
                        except Exception as e:
                            with open(BatchLog, 'a') as f:
                                f.write('\n \t Failed for case: '+ subfolders[i])

                            print('Exception for case:', subfolders[i] )



if __name__ == '__main__':


   
    startLDT = time.time()
 
    #psutil.cpu_count()
    #pid = os.getpid()
    #os.system("taskset -cp %d %d" %(65, pid))
    p = psutil.Process()
    #p.cpu_affinity(list(range(workers))) 


    main_path = '.'          ### folder where the run_parallel code is located
    model_path = './model'
    cases_folder = os.path.join(main_path,'test_data')
    cases_list = [os.path.join(cases_folder, batch) for batch in os.listdir(cases_folder)]
    # acquire file containing the case_id and ED/ES frames
    # comment next line if this file is not available
    file_CaseFrame = 'Case_ID_and_frame_44k.csv'
    
    case_frame_dict = None
    try: 
        with open(file_CaseFrame , 'r') as f:
            Lines = f.readlines()
            list_lines = []
            for i, line in enumerate(Lines):
                list_lines.append(line.strip())
        
        case_frame_dict = {}
        # build a dictionary with structure: 'case': [ED frame, ES frame]
        for i in list_lines[1:]: # skip header
            i = i.split(',')
            case_frame_dict[i[0]] = [i[1], i[2]]
    except: 
        pass

    if case_frame_dict is not None:
        results = split_and_run(cases_list, workers)
    else: 
        results = split_and_run(cases_list, workers, id_Frame = case_frame_dict)

    print('TOTAL TIME: ', time.time()-startLDT)

    #results =[perform_fitting(folder, iter_num = i) for i,folder in enumerate(cases_folders)]



