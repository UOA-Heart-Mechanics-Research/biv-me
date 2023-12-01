import os
import numpy as np
import time
import pandas as pd 
import re
from plotly.offline import  plot
import plotly.graph_objs as go
import sys

from bivme.fitting.GPDataSet import *


case_folder='R:\\resmed201900006-biomechanics-in-heart-disease\\Sandboxes\\Debbie\\collaborations\\chicago-rv-mesh\\analysis\\gpfiles-raw\\RV01' # this is the folder where I have stored your GP and SliceInfoFile

suffix = ''


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

    frameID = 'ED'
    frameID = 'def'  
    
    GPfilename = os.path.join(folder, f'GPfile{suffix}.txt') 
    filenameInfo = os.path.join(folder, 'SliceInfoFile.txt')

    # this section is only used to measure the shift
    time_frame_ED = 0 # choose the correct time frame number
    # ED_dataset = GPDataSet(os.path.join(folder, 'GP_ED.txt'),filenameInfo, case, sampling = 1, time_frame_number = time_frame_ED)
    ED_dataset = GPDataSet(GPfilename,filenameInfo, case, sampling = 1, time_frame_number = time_frame_ED)
    result_ED = ED_dataset.sinclaire_slice_shifting( frame_num = time_frame_ED) 
    shift_ED = result_ED[0]
    pos_ED = result_ED[1]   
            
    contours_to_plot = [
                                ContourType.SAX_RV_FREEWALL, ContourType.LAX_RV_FREEWALL,
                                ContourType.SAX_RV_SEPTUM, ContourType.LAX_RV_SEPTUM,
                                ContourType.SAX_LV_ENDOCARDIAL, ContourType.SAX_RV_ENDOCARDIAL,
                                ContourType.SAX_LV_EPICARDIAL, ContourType.RV_INSERT,
                                ContourType.APEX_POINT, ContourType.MITRAL_VALVE,
                                ContourType.TRICUSPID_VALVE,
                                ContourType.SAX_RV_EPICARDIAL, ContourType.LAX_RV_EPICARDIAL, ContourType.LAX_RV_ENDOCARDIAL, 
                                ContourType.LAX_LV_ENDOCARDIAL, ContourType.LAX_LV_EPICARDIAL,
                                ContourType.SAX_RV_OUTLET,
                                ContourType.PULMONARY_PHANTOM, ContourType.AORTA_VALVE,
                                ContourType.PULMONARY_VALVE, ContourType.TRICUSPID_PHANTOM,
                                ContourType.AORTA_PHANTOM, ContourType.MITRAL_PHANTOM, ContourType.LAX_LV_EXTENT,ContourType.LAX_RV_EXTENT
                                ]

    # load GP for chosen frame and apply shift measured at ED
    time_frame_GP = 0
    data_set = GPDataSet(GPfilename,filenameInfo, case, 1, time_frame_number = time_frame_GP) #18 is ES (RV) (timeframe 0-49)
    data_set.apply_slice_shift(shift_ED, pos_ED)
    contourPlots = data_set.PlotDataSet(contours_to_plot)

    print('Frame ', frameID , ' done')

    # plot hmtl time series for all chosen frames
    plot(go.Figure(contourPlots),filename=os.path.join(folder,
                                           f'GuidePoints_{suffix}_{time_frame_GP}.html'), auto_open=False)
                

if __name__ == '__main__':

    
    startLDT = time.time()
   
    # results = Plot_html('./venus1')
    results = Plot_html(case_folder)
    print('TOTAL TIME: ', time.time()-startLDT)
