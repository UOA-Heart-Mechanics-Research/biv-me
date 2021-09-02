#!/usr/bin/env python3
import os
os.system('CLS')
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import tools
import importlib
import math
import sys
from functools import partial
from IPython.display import display
import biventricularModel
importlib.reload(biventricularModel)
from biventricularModel import *
import numpy as np
import pandas as pd
import GPDataSet
importlib.reload(GPDataSet)
from GPDataSet import*
import copy
import Error
importlib.reload(Error)
from Error import*
init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings("ignore")
import numpy
import cvxopt


# file containing 3D points, labels, slice number and time frame
case = 'AKL'
filename = './TestHao/GPFileAKL.txt'            # file containing 3D points, labels, slice number and time frame
filenameInfo = './TestHao/SliceInfoFile.txt'  # file containing dicom info - not needed here

# Path to your working directory
saving_path = './'                                       # this is where the results are saved

DataSet = GPDataSet(filename,filenameInfo,case,1)

# Loads biventricular control_mesh
control_mesh = (pd.read_table("./model.txt",delim_whitespace=True,header=None)).values

biventricular_model = biventricularModel(control_mesh,DataSet,case,filenameInfo)

scale = biventricular_model.UpdatePoseAndScale()

[Mitral_phantom,Mitral,Tricuspid_phantom,Tricuspid,Pulmonary_phantom,Pulmonary,Aorta_phantom,Aorta,RV,RVS,LV,Epi,RVinsert,Apex] = DataSet.PlotDataSetEvenlySpaced()

Model = biventricular_model.PlotSurface("rgb(0,127,0)","rgb(0,0,127)","rgb(127,0,0)","Initial model","all")

data = Model+ Mitral_phantom+Mitral+Tricuspid_phantom+Tricuspid+Pulmonary_phantom+Pulmonary+Aorta_phantom+Aorta+RV+RVS+LV+Epi+RVinsert+Apex
iplot(go.Figure(data= go.Data(data)))

weight_GP = 40
biventricular_model.MultiThreadSmoothingED(case,weight_GP)

# Results
Model = biventricular_model.PlotSurface("rgb(0,127,0)","rgb(0,0,127)","rgb(127,0,0)","Initial model","all")
[Mitral_phantom,Mitral,Tricuspid_phantom,Tricuspid,Pulmonary_phantom,Pulmonary,Aorta_phantom,Aorta,RV,RVS,LV,Epi,RVinsert,Apex] = biventricular_model.DataSet.PlotDataSetEvenlySpaced()

data = Model + Mitral_phantom+Mitral+Tricuspid_phantom+Tricuspid+Pulmonary_phantom+Pulmonary+Aorta_phantom+Aorta+RV+RVS+LV+Epi+RVinsert+Apex
iplot(go.Figure(data= go.Data(data)))

transmural_weight = 0.01
low_smoothing_weight = 1000
biventricular_model.SolveProblemCVXOPT(case,weight_GP,low_smoothing_weight,transmural_weight)

# Results
ModelEndo = biventricular_model.PlotSurface("rgb(0,127,0)","rgb(0,0,127)","rgb(127,0,0)","Initial model","endo")
ModelEpi = biventricular_model.PlotSurface("rgb(0,127,0)","rgb(0,0,127)","rgb(127,0,0)","Initial model","epi")
Model = biventricular_model.PlotSurface("rgb(0,127,0)","rgb(0,0,127)","rgb(127,0,0)","Initial model","all")

[Mitral_phantom,Mitral,Tricuspid_phantom,Tricuspid,Pulmonary_phantom,Pulmonary,Aorta_phantom,Aorta,RV,RVS,LV,Epi,RVinsert,Apex] = biventricular_model.DataSet.PlotDataSetEvenlySpaced()

data = ModelEndo + Mitral+Tricuspid+Pulmonary+Aorta+RV+RVS+LV+RVinsert+Apex
iplot(go.Figure(data= go.Data(data)))

data = ModelEpi + Mitral+Tricuspid+Pulmonary+Aorta+Epi+Apex
iplot(go.Figure(data= go.Data(data)))

data = Model + Mitral+Tricuspid+Pulmonary+Aorta+RV+RVS+LV+RVinsert+Apex+Epi
iplot(go.Figure(data= go.Data(data)))

# save the results in stl
x = np.column_stack((biventricular_model.control_mesh[:, 0],
                     biventricular_model.control_mesh[:, 1],
                     biventricular_model.control_mesh[:, 2]))
np.savetxt(os.path.join('./TestHao/' + case + '_control_mesh.txt'), x)
