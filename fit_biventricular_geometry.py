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

path = '/home/am20/dev/BiVFitting'
case = 'test_data'


filename = os.path.join(path, case, 'GPFile.txt')  # file
# labels, slice number and time frame
filenameInfo = os.path.join(path, case,'SliceInfoFile.txt')  #
# file containing dicom info
saving_path = '/home/am20/dev/storage/models'  # this is where the
# results are saved

weight_GP = 200
low_smoothing_weight = 500
transmural_weight = 500


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
                    ContourType.PULMONARY_VALVE, ContourType.AORTA_VALVE
                    ]


# Loads DataSet.
# Load ED frame
data_set = GPDataSet(filename,filenameInfo, case, 1)

data_set.sinclaire_slice_shifting()
# data_set.Identify_RVS_LAX()

# Loads biventricular control_mesh
model_path = "/home/am20/dev/BiVFitting/model"
biventricular_model = BiventricularModel(model_path, case)
biventricular_model.update_pose_and_scale(data_set)

# # perform a stiff fit
# displacement, err = biventricular_model.lls_fit_model(weight_GP,data_set,1e10)
# biventricular_model.control_mesh = np.add(biventricular_model.control_mesh,
#                                           displacement)
# biventricular_model.et_pos = np.linalg.multi_dot([biventricular_model.matrix,
#                                                   biventricular_model.control_mesh])
# displacements = data_set.SAXSliceShiffting(biventricular_model)

contourPlots = data_set.PlotDataSet(contours_to_plot)
model = biventricular_model.plot_surface("rgb(0,127,0)",
                                         "rgb(0,0,127)",
                                         "rgb(127,0,0)",
                                         surface = "all")

#plot(go.Figure(contourPlots))
data = contourPlots
plot(go.Figure(data),filename=os.path.join(path,case,
                                           'pose_fitted_model.html'),
                 auto_open=False)



# Generates RV epicardial point if they have not been contoured
# (can be commented if available)
rv_epi_points,rv_epi_contour, rv_epi_slice = data_set.create_rv_epicardium(
    rv_thickness=3)


# Generates 30 BP_point phantom points and 30 tricuspid phantom points.
# We do not have any pulmonary points or aortic points in our dataset but if you do,
# I recommend you to do the same.

mitral_points = data_set.create_valve_phantom_points(30, ContourType.MITRAL_VALVE)
tri_points = data_set.create_valve_phantom_points(30, ContourType.TRICUSPID_VALVE)
pulmonary_points = data_set.create_valve_phantom_points(20, ContourType.PULMONARY_VALVE)


# Exmple on how to set different weights for different points group
data_set.weights[
    data_set.contour_type == ContourType.APEX_POINT] \
    = 5
data_set.weights[
    data_set.contour_type == ContourType.MITRAL_VALVE] \
    = 5
data_set.weights[
    data_set.contour_type == ContourType.TRICUSPID_VALVE] \
    = 5
data_set.weights[
    data_set.contour_type == ContourType.AORTA_VALVE] \
    = 5
data_set.weights[
    data_set.contour_type == ContourType.PULMONARY_VALVE] \
    = 5
data_set.weights[
    data_set.contour_type == ContourType.RV_INSERT] \
    = 5

MultiThreadSmoothingED(biventricular_model,weight_GP, data_set)


# Results
model = biventricular_model.plot_surface("rgb(0,127,0)", "rgb(0,0,127)",
                            "rgb(127,0,0)","all")
data = model + contourPlots
plot(go.Figure(data),filename=os.path.join(path, case,
                                           'step1_fitted_model.html'),
                 auto_open=False)

SolveProblemCVXOPT(biventricular_model,data_set,weight_GP,low_smoothing_weight,
                                       transmural_weight)



# Results
model = biventricular_model.plot_surface("rgb(0,127,0)", "rgb(0,0,127)",
                            "rgb(127,0,0)","all")
data = model + contourPlots
plot(go.Figure(data),filename=os.path.join(path, case,
                                           'step2_fitted_model.html'),
                 auto_open=False)



# Save
# x = np.column_stack((surface.control_mesh[:,0],surface.control_mesh[:,1],surface.control_mesh[:,2]))
# np.savetxt(str(saving_path)+str(case)+'/'+str(case)+'_Model_file_diffeo_'+str(1)+'.txt', x)







