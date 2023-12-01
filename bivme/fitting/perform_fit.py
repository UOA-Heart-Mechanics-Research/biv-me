# Input: 3D contours
# Output: Fitted model

#!/usr/bin/env python3
import os
from plotly.offline import  plot
import plotly.graph_objs as go
import numpy as np
from BiVFitting import BiventricularModel
from BiVFitting import GPDataSet
from BiVFitting import ContourType
from BiVFitting import MultiThreadSmoothingED, SolveProblemCVXOPT
from BiVFitting import plot_timeseries
import time
import pandas as pd 
from pathlib import Path

from config_params import * 

#This list of contours_to _plot was taken from Liandong Lee
contours_to_plot = [ContourType.LAX_RA, ContourType.LAX_RV_ENDOCARDIAL,
                    ContourType.SAX_RV_FREEWALL, ContourType.LAX_RV_FREEWALL,
                    ContourType.SAX_RV_SEPTUM, ContourType.LAX_RV_SEPTUM,
                    ContourType.SAX_LV_ENDOCARDIAL,
                    ContourType.SAX_LV_EPICARDIAL, ContourType.RV_INSERT,
                    ContourType.APEX_POINT, ContourType.MITRAL_VALVE,
                    ContourType.TRICUSPID_VALVE, ContourType.AORTA_VALVE,
                    ContourType.SAX_RV_EPICARDIAL, ContourType.LAX_RV_EPICARDIAL,
                    ContourType.LAX_LV_ENDOCARDIAL, ContourType.LAX_LV_EPICARDIAL,
                    ContourType.LAX_RV_EPICARDIAL, ContourType.SAX_RV_OUTLET,
                    ContourType.AORTA_PHANTOM, ContourType.TRICUSPID_PHANTOM,
                    ContourType.MITRAL_PHANTOM
                    ]

def perform_fitting(folder, **kwargs):
    #performs all the BiVentricular fitting operations

    try:
              
        if 'iter_num' in kwargs:
            iter_num = kwargs.get('iter_num', None)
            pid = os.getpid()
            #print('child PID', pid)
            # assign a new process ID and a new CPU to the child process 
            # iter_num corresponds to the id number of the CPU where the process will be run
            os.system("taskset -cp %d %d" %(iter_num, pid))

        if 'id_Frame' in kwargs:
            # acquire .csv file containing patient_id, ES frame number, ED frame number if present
            case_frame_dict = kwargs.get('id_Frame', None)

        # define the path to GPFile and to SliceInfoFile
        # THIS SHOULD BE CHANGED if files are named differently
        filename = os.path.join(folder, 'GPFile.txt') 
        filenameInfo = os.path.join(folder,'SliceInfoFile.txt')

        # extract the patient name from the folder name
        case =  os.path.basename(os.path.normpath(folder))
        print('case: ', case )

        #read all the frames from the GPFile 
        all_frames = pd.read_csv(filename, sep = '\t')
        # select which frames to fit
        ED_frame = None
        try:
            ED_frame = int(case_frame_dict[str(case)][0])
        except: 
            ED_frame = 0
            print('ED set to frame # 0')
        
        #frames_to_fit = np.array(case_frame_dict[str(case)]) # oly fit ED and ES, if ED_ES file provided
        frames_to_fit = np.unique([i[6] for i in all_frames.values]) # if you want to fit all _frames


        # create a separate output folder for each patient
        model_path = './model'
        output_folder = './results/' + case
        try:
            os.makedirs(output_folder , exist_ok= True)
        except: raise ValueError

        # create log Files where to store fitting errors and shift
        Errorfile = Path(os.path.join(output_folder ,'ErrorFile.txt'))
        Errorfile.touch(exist_ok=True)
        Shiftfile = Path(os.path.join(output_folder ,'Shiftfile.txt'))
        Shiftfile.touch(exist_ok=True)

        Posfile = Path(os.path.join(output_folder ,'Posfile.txt'))	
        Posfile.touch(exist_ok=True)	

        with open(Errorfile, 'w') as f:
            f.write('Log for patient: '+ case+'\n')    


        # The next lines are used to measure shift using only a key frame
        if measure_shift_EDonly == True:

            print('shift measured only at ED frame')

            ED_dataset = GPDataSet(filename,filenameInfo, case, sampling = sampling, time_frame_number = ED_frame)
            result_ED = ED_dataset.sinclaire_slice_shifting( frame_num = ED_frame) 
            shift_ED = result_ED[0]
            pos_ED = result_ED[1]
            #np.save(os.path.join(output_folder, 'shift.txt'), shift_ED)
            with  open(Shiftfile, "w") as file:
                    file.write('shift measured only at ED: frame '+ str(ED_frame)+'\n')
                    file.write(str(shift_ED))
                    file.close()        

            with  open(Posfile, "w") as file:	
                    file.write('pos measured only at ED: frame '+ str(ED_frame)+'\n')	
                    file.write(str(pos_ED))	
                    file.close()

        #initialise time series lists
        TimeSeries_step1 = []
        TimeSeries_step2 = []
	
        print('FITTING OF  ', str(case), '----> started \n')

        for idx,num in enumerate(sorted(frames_to_fit)):
                num = int(num) #frame number
                print('frame num', num)

                Modelfile = Path(os.path.join(output_folder , str(case)+'_Model_Frame_'+"{0:03}".format(num)+'.txt'))
                Modelfile.touch(exist_ok=True)  

                with open(Errorfile, 'a') as f: 
                    f.write('\nFRAME #' +str(int(num))+'\n')

                data_set = GPDataSet(filename,filenameInfo, case, sampling = sampling, time_frame_number = num)
                biventricular_model = BiventricularModel(model_path, case)                        
                model = biventricular_model.plot_surface("rgb(0,127,0)",  
                                                        "rgb(0,0,127)",
                                                        "rgb(127,0,0)",
                                                        surface = "all") 


                if measure_shift_EDonly == True:
                    # apply shift measured previously using ED frame
                    data_set.apply_slice_shift(shift_ED, pos_ED)
                    data_set.get_unintersected_slices()
                else: 
                    # measure and apply shift to current frame
                    shiftedSlice = data_set.sinclaire_slice_shifting(Errorfile, int(num)) 
                    shiftmeasure = shiftedSlice[0]
                    posmeasure = shiftedSlice[1]

                    if idx == 0:  
                        with  open(Shiftfile, "w") as file:
                            file.write('Frame number:  ' + str(num)+'\n')
                            file.write(str(shiftmeasure))
                            file.close()  
                        ]
                        with  open(Posfile, "w") as file:	
                            file.write('Frame number:  ' + str(num)+'\n')	
                            file.write(str(posmeasure))	
                            file.close()    

                    else:  
                        with  open(Shiftfile, "a") as file:
                            file.write('\nFrame number:  ' + str(num)+'\n')
                            file.write(str(shiftmeasure))
                            file.close()    

                        with  open(Posfile, "w") as file:	
                            file.write('\nFrame number:  '+ str(num)+'\n')	
                            file.write(str(posmeasure))	
                            file.close()    
                    pass 
                
                #model_path = "./model"

                biventricular_model.update_pose_and_scale(data_set)

                # # perform a stiff fit
                # displacement, err = biventricular_model.lls_fit_model(weight_GP,data_set,1e10)
                # biventricular_model.control_mesh = np.add(biventricular_model.control_mesh,
                #                                           displacement)
                # biventricular_model.et_pos = np.linalg.multi_dot([biventricular_model.matrix,
                #                                                   biventricular_model.control_mesh])
                # displacements = data_set.SAXSliceShiffting(biventricular_model)

                contourPlots = data_set.PlotDataSet(contours_to_plot)    


                #plot(go.Figure(contourPlots))
                data = contourPlots

                #plot(go.Figure(data),filename=os.path.join(folder, 'pose_fitted_model_Frame'+str(int(num))+'.html'), auto_open=False) 

                # Generates RV epicardial point if they have not been contoured
                # (can be commented if available) used in LL 
                rv_epi_points,rv_epi_contour, rv_epi_slice = data_set.create_rv_epicardium(
                    rv_thickness=3)


                # Generates 30 BP_point phantom points and 30 tricuspid phantom points.
                # We do not have any pulmonary points or aortic points in our dataset but if you do,
                # I recommend you to do the same.

                mitral_points = data_set.create_valve_phantom_points(30, ContourType.MITRAL_VALVE)
                tri_points = data_set.create_valve_phantom_points(30, ContourType.TRICUSPID_VALVE)
                pulmonary_points = data_set.create_valve_phantom_points(20, ContourType.PULMONARY_VALVE)
                aorta_points = data_set.create_valve_phantom_points(20, ContourType.AORTA_VALVE)
            

                # Example on how to set different weights for different points group (R.B.)
                data_set.weights[data_set.contour_type == ContourType.MITRAL_PHANTOM] = 2
                data_set.weights[data_set.contour_type == ContourType.AORTA_PHANTOM] = 2
                data_set.weights[data_set.contour_type == ContourType.PULMONARY_PHANTOM] = 2
                data_set.weights[data_set.contour_type == ContourType.TRICUSPID_PHANTOM] = 2

                data_set.weights[data_set.contour_type == ContourType.APEX_POINT] = 1
                data_set.weights[data_set.contour_type == ContourType.RV_INSERT] = 5
                
                data_set.weights[data_set.contour_type == ContourType.MITRAL_VALVE] = 2
                data_set.weights[data_set.contour_type == ContourType.AORTA_VALVE]= 2
                data_set.weights[data_set.contour_type == ContourType.PULMONARY_VALVE] = 2                
                
                # Perform linear fit
                MultiThreadSmoothingED(biventricular_model,weight_GP, data_set, Errorfile)
 
                # Plot results
                model = biventricular_model.plot_surface("rgb(0,127,0)", "rgb(0,0,127)","rgb(127,0,0)","all")   
                data = model + contourPlots
                #TimeSeries_step1.append([data, num])

                # Perform diffeomorphic fit
                SolveProblemCVXOPT(biventricular_model,data_set,weight_GP,low_smoothing_weight,
                                                    transmural_weight,Errorfile)

                # Plot final results
                model = biventricular_model.plot_surface("rgb(0,127,0)", "rgb(0,0,127)", "rgb(127,0,0)","all")
                data = model + contourPlots
                #TimeSeries_step2.append([data, num])
                
                #plot(go.Figure(data),filename=os.path.join(folder, 'step2_fitted_model_Frame'+str(int(num))+'.html'), auto_open=False)
                
                # save results in .txt format, one file for each frame
                ModelData = {'x': biventricular_model.control_mesh[:,0], 'y': biventricular_model.control_mesh[
                    :,1], 'z': biventricular_model.control_mesh[:,2], 'Frame': [num] * len(biventricular_model.control_mesh[:,2])}
                
                Model_Dataframe = pd.DataFrame(data=ModelData)
                with  open(Modelfile, "w") as file:
                    file.write(Model_Dataframe.to_csv(header=True, index=False, sep = ',', line_terminator='\n'))

        # if you want to plot time series in html files uncomment the next line(s)
        #plot_timeseries(TimeSeries_step1, output_folder, 'TimeSeries_step1.html')
        #plot_timeseries(TimeSeries_step2, output_folder, 'TimeSeries_step2.html')
	
        DoneFile = Path(os.path.join(output_folder ,'Done.txt'))
        DoneFile.touch(exist_ok=True)   
	
    except KeyboardInterrupt:
        raise KeyboardInterruptError()




if __name__ == '__main__':

    
    startLDT = time.time()
    #pid = os.getpid()
    #os.system("taskset -cp %d %d" %(66, pid))

    main_path = '.'          ### folder in use

    cases_folder = os.path.join(main_path, 'test_data')
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
        results = [perform_fitting (folder, id_Frame = case_frame_dict) for folder in cases_list]
    else:
        results = [perform_fitting (folder) for folder in cases_list]

    print('TOT CASES:', len(cases_list))
    
    print('TOTAL TIME: ', time.time()-startLDT)

