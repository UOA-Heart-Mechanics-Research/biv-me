# Created by LDT on 14 April 2022
# This script contains functions to create and save vtk files, starting from the txt file containing the coarse mesh coordinates 


#!/usr/bin/env python3
import os
import numpy as np
import time
import pandas as pd 
import re

import sys
sys.path.append( '../../BiV_Modelling' ) # append path to the Fitting framework where the BiVFitting folder is located

from BiVFitting.BiventricularModel import BiventricularModel
from BiVFitting.GPDataSet import GPDataSet
from BiVFitting.Diffeomorphic_fitting import plot_timeseries
from BiVFitting.surface_enum import ContourType


def Plot_html(folder, **kwargs):
    '''
    Author: ldt 
    Date: 26/05/2022
    -------------------------
    Input: 
    - folder: is the folder where the txt file containing the coarse mesh coordinates is saved
    - test_data_folder: is the fodler where the GPFile.txt files are saved. This is optional, 
                        if not provided the guide points will not be in the output html file.
    -------------------------
    Output: 
    - html file that shows the mesh (and the guide points, only if given as input).

    '''

    # extract case name
    case =  os.path.basename(os.path.normpath(folder))
    print('case', case)

    # look for the guide point files that correspond to the mesh. This input is optional.
    if 'test_data_folder' in kwargs:
        test_data_folder = kwargs.get('test_data_folder', None)
        GPfilename = os.path.join(test_data_folder, case+'/GPFile.txt')  #GPfilename = os.path.join(test_data_folder, case+'/GPFile_ldt.txt') 
        filenameInfo = os.path.join(test_data_folder, case+'/SliceInfoFile.txt') #filenameInfo = os.path.join(test_data_folder, case+'/SliceInfoFile_ldt.txt')


    # find and sort the txt files where the output mesh has been saved
    ModelData = [filename for filename in os.listdir(folder) if 'Model_Frame_' in filename]
    ModelData = sorted(ModelData)

    if 'test_data_folder' in kwargs:
        if test_data_folder != None:
            # load the guide points, measure slice shift at ED
            ED_dataset = GPDataSet(GPfilename ,filenameInfo, case, sampling = 1, time_frame_number = 1)
            result_ED = ED_dataset.sinclaire_slice_shifting( frame_num = 0) 
            shift_ED = result_ED[0]
            pos_ED = result_ED[1]   

    Time_series = []
    for file in ModelData:

            frame_ID = int((re.search('Frame_(\d+)', file)).group(1))

            #model_path = '../Fitting_framework/model' # this is wrong
            model_path = '../../BiV_Modelling/model' # this is wront
            #filename = fp #+ '/GPFile_proc.txt' #only proc for old
            #filenameInfo = '/home/rb20/Desktop/CharleneModelsRVLVPaper/Models/New_patient_position' + case_ID + '.txt'#fp + '/SliceInfoFile_proc.txt'

            #load biventricular model
            shifting_model = BiventricularModel(model_path,os.path.join(folder, file ))
            #print(os.path.join(folder,file))
            fitted_nodes = (pd.read_csv(os.path.join(
                folder, file), sep=",", skiprows=0 , dtype = np.float64)).values
            shifting_model.update_control_mesh(fitted_nodes)
            
            contours_to_plot = [ContourType.LAX_RA,
                                ContourType.SAX_RV_FREEWALL, ContourType.LAX_RV_FREEWALL,
                                ContourType.SAX_RV_SEPTUM, ContourType.LAX_RV_SEPTUM,
                                ContourType.SAX_LV_ENDOCARDIAL,
                                ContourType.SAX_LV_EPICARDIAL, ContourType.RV_INSERT,
                                ContourType.APEX_POINT, ContourType.MITRAL_VALVE,
                                ContourType.TRICUSPID_VALVE,
                                ContourType.SAX_RV_EPICARDIAL, ContourType.LAX_RV_EPICARDIAL,
                                ContourType.LAX_LV_ENDOCARDIAL, ContourType.LAX_LV_EPICARDIAL,
                                ContourType.LAX_RV_EPICARDIAL, ContourType.SAX_RV_OUTLET,
                                ContourType.PULMONARY_PHANTOM, ContourType.AORTA_VALVE,
                                ContourType.PULMONARY_VALVE, ContourType.TRICUSPID_PHANTOM,
                                ContourType.AORTA_PHANTOM, ContourType.MITRAL_PHANTOM
                                ]

            if 'test_data_folder' in kwargs:
                if test_data_folder != None:
                    # load GP for current frame and apply shift measured at ED
                    data_set = GPDataSet(GPfilename,filenameInfo, case, 1, time_frame_number = frame_ID) #18 is ES (RV) (timeframe 0-49)
                    data_set.apply_slice_shift(shift_ED, pos_ED)
                    contourPlots = data_set.PlotDataSet(contours_to_plot)
                    model = shifting_model.PlotSurface("rgb(0,127,0)", "rgb(0,0,127)", "rgb(127,0,0)","Initial model", "all")
                    data = contourPlots + model

            else: 
                data = model

            print('Frame ', frame_ID , ' done')
            Time_series.append([data, frame_ID])
    
    # plot hmtl time series for all chosen frames
    plot_timeseries(Time_series,folder, 'TimeSeries_FromModel.html')


if __name__ == '__main__':

    
    startLDT = time.time()

    #main_path = 'C:/Users/ldt18/Desktop/Dev_BioBank' # fix this  
    main_path_pre='C:/Users/an11/Desktop/Work/Repositories/YoungLabBivModelling'; # this is the path to the folder where the Biv_Modelling folder is saved.
    main_path = os.path.join(main_path_pre,'BiV_Modelling') # fix this    "C:\Users\an11\Desktop\Work\Repositories\YoungLabBivModelling\BiVModelling_tools-main\BiVModelling_tools-main\Plotting\Plot_html.py"    
    cases_folder = os.path.join(main_path, 'results') # path where patient folders containing .txt models are located
    cases_list = [os.path.join(cases_folder, batch) for batch in os.listdir(cases_folder) ]
    #results = [ Plot_html(folder, test_data_folder = 'C:/Users/an11/Desktop/Work/Repositories/YoungLabBivModelling/BiV_Modelling/test_data') for folder in cases_list]
    results = [ Plot_html(folder, test_data_folder = os.path.join(main_path,'test_data')) for folder in cases_list]
    print('TOTAL TIME: ', time.time()-startLDT)

