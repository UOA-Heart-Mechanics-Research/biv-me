import os
import numpy as np
import time
import pandas as pd
import re
from plotly.offline import plot
import plotly.graph_objs as go
import sys

from bivme.fitting.GPDataSet import *


def generate_html(folder, gpsuffix='', frame=0):
    """
    Generates an HTML file containing plots of guide points.

    Parameters:
    folder (str): The path to the folder containing the guide point files.
    gpsuffix (str): The suffix to be added to the guide point file name.
    frame (int, optional): The frame number to generate the plots for. Defaults to 0.
    """

    # extract case name
    case = os.path.basename(os.path.normpath(folder))
    print("case", case)

    GPfilename = os.path.join(folder, f"GPfile{gpsuffix}.txt")
    filenameInfo = os.path.join(folder, "SliceInfoFile.txt")

    # this section is only used to measure the shift
    time_frame_ED = 0 # this is the ED frame
    ED_dataset = GPDataSet(
        GPfilename, filenameInfo, case, sampling=1, time_frame_number=time_frame_ED
    )
    result_ED = ED_dataset.sinclaire_slice_shifting(frame_num=time_frame_ED)
    shift_ED = result_ED[0]
    pos_ED = result_ED[1]

    contours_to_plot = [
        ContourType.SAX_RV_FREEWALL,
        ContourType.LAX_RV_FREEWALL,
        ContourType.SAX_RV_SEPTUM,
        ContourType.LAX_RV_SEPTUM,
        ContourType.SAX_LV_ENDOCARDIAL,
        ContourType.SAX_RV_ENDOCARDIAL,
        ContourType.SAX_LV_EPICARDIAL,
        ContourType.RV_INSERT,
        ContourType.APEX_POINT,
        ContourType.MITRAL_VALVE,
        ContourType.TRICUSPID_VALVE,
        ContourType.SAX_RV_EPICARDIAL,
        ContourType.LAX_RV_EPICARDIAL,
        ContourType.LAX_RV_ENDOCARDIAL,
        ContourType.LAX_LV_ENDOCARDIAL,
        ContourType.LAX_LV_EPICARDIAL,
        ContourType.SAX_RV_OUTLET,
        ContourType.PULMONARY_PHANTOM,
        ContourType.AORTA_VALVE,
        ContourType.PULMONARY_VALVE,
        ContourType.TRICUSPID_PHANTOM,
        ContourType.AORTA_PHANTOM,
        ContourType.MITRAL_PHANTOM,
        ContourType.LAX_LV_EXTENT,
        ContourType.LAX_RV_EXTENT,
    ]

    # load GP for chosen frame and apply shift measured at ED
    time_frame_GP = frame
    data_set = GPDataSet(
        GPfilename, filenameInfo, case, 1, time_frame_number=time_frame_GP
    ) 
    data_set.apply_slice_shift(shift_ED, pos_ED)
    contourPlots = data_set.PlotDataSet(contours_to_plot)

    print(f"Guidepoints plotted for frame {frame}")

    # plot hmtl time series for all chosen frames
    plot(
        go.Figure(contourPlots),
        filename=os.path.join(folder, f"{case}_guidepoints{gpsuffix}_frame_{time_frame_GP}.html"),
        auto_open=False,
    )


if __name__ == "__main__":
    
    # folder containing GPFile.txt and SliceInfoFile.txt
    case_folder = r"R:\resmed201900006-biomechanics-in-heart-disease\Sandboxes\Debbie\collaborations\chicago-rv-mesh\analysis\gpfiles-raw\RV01"
    gpsuffix = ''

    starttime = time.time()
    results = generate_html(case_folder, gpsuffix)

    # print time taken to 1 decimal place
    print(f"Time taken: {time.time() - starttime:.1f} seconds")
