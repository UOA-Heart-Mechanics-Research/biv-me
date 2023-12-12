import os
import glob

# Reads valve points from cim rvlv models using keys
# Author: Joshua Dillon
# Last Modified: December, 2023

## Keys

GP_KEYS = {'GP_APEX_CENTRAL_AXIS': 0,
  'GP_BASE_CENTRAL_AXIS': 1,
  'GP_RV_INSERTION': 2,
  'GP_BASE_PLANE': 3,
  'GP_INTERPOLATED_BASEPOINT': 4,
  'GP_LV_ENDO': 5,
  'GP_LV_EPI': 6,
  'GP_LV_ENDO_CONSTRAINED': 7,
  'GP_LV_EPI_CONSTRAINED': 8,
  'GP_LV_ENDO_IMAGE': 9,
  'GP_LV_EPI_IMAGE': 10,
  'GP_GEOM_PARAM': 11,
  'GP_CENTROID': 12,
  'GP_MYO2D': 13,
  'GP_MYO1D': 14,
  'GP_LV_BASE_ENDO': 15,
  'GP_LV_BASE_EPI': 16,
  'GP_MYOIMAGE1D': 17,
  'GP_RVFW_ENDO': 18,  
  'GP_RVS_ENDO': 19,
  'GP_EPI': 20,
  'GP_VALVE_MITRAL': 21,
  'GP_VALVE_AORTIC': 22,
  'GP_VALVE_TRICUSPID': 23,
  'GP_VALVE_PULMONARY': 24,
  'GP_RV_CENTROID': 25,  
  'GP_RVFW_ENDO_IMAGE': 26,  
  'GP_RVS_ENDO_IMAGE': 27,
  'GP_EPI_IMAGE': 28,
  'GP_ROI': 29,
  'GP_LV_EPI_WARP_CONTOUR': 30, 
  'GP_LV_ENDO_WARP_CONTOUR': 31,
  'GP_MRSEPTAL': 32, 
  'GP_MRSEPTAL_INTERPOLATED': 33,
  'GP_MRLATERAL': 34,
  'GP_MRLATERAL_INTERPOLATED': 35,
  'GP_MITRAL_CENTROID': 36, 
  'GP_AORTIC_CENTROID': 37,
  'GP_TRICUSPID_CENTROID': 38,
  'GP_PULMONARY_CENTROID': 39,
  'GP_LV_EPI_APEX': 40,
  'GP_RVFW_ENDO_WARP_CONTOUR': 41,
  'GP_RVS_ENDO_WARP_CONTOUR': 42,
  'GP_EPI_WARP_CONTOUR': 43,
  'GP_VALVE_MITRAL_WARP_CONTOUR': 44,
  'GP_VALVE_AORTIC_WARP_CONTOUR': 45,
  'GP_VALVE_TRICUSPID_WARP_CONTOUR': 46,
  'GP_VALVE_PULMONARY_WARP_CONTOUR': 47,
  'GP_RV_INSERTION_WARP': 48, 
  'GP_VALVE_MITRAL_WARP': 49, 
  'GP_VALVE_AORTIC_WARP': 50,
  'GP_VALVE_TRICUSPID_WARP': 51,
  'GP_VALVE_PULMONARY_WARP': 52,
  'GP_LV_EPI_APEX_WARP': 53,  
  'PP_VALVE_MITRAL': 54, 
  'PP_VALVE_AORTIC': 55, 
  'PP_VALVE_TRICUSPID': 56,
  'PP_VALVE_PULMONARY': 57,
  'PP_LV_ENDO': 58, 
  'PP_RVFW_ENDO': 59,
  'PP_RVS_ENDO': 60,
  'PP_EPI': 61,
  'NUM_GPDATA_TYPE': 62,
}

# Read guidepoints file
def read_guidepoints(caseID, analystID, rvlvpath):
    rvlvpath = os.path.join(rvlvpath, caseID)
    # Get guidepoints file
    guidepoints_file = glob.glob(rvlvpath+'/'+'*'+analystID+'/'+'guide_points.data')
    if len(guidepoints_file) == 0:
        print('No guidepoints file found for case ' + caseID)
        return None
    guidepoints_file = guidepoints_file[0]
    
    # Read guidepoints file
    with open(guidepoints_file) as f:
        lines = f.readlines()
    
    # Parse guidepoints file
    guidepoints = [{}]
    slice_num = []
    index = 0
    for line in lines:
        index += 1
        line = line.split()
        if line[0] == 'number':
            pass
        else:
            gp_key = line[3]
            gp_type = [k for k, v in GP_KEYS.items() if str(v) == gp_key]
            if gp_type == []:
                gp_type = 'unknown'
            else:
                guidepoints.append({'name': gp_type[0], 'x': line[0], 'y': line[1], 'z': line[2], 'series': line[5], 'slice': line[6], 'frame': line[7]})
                slice_num.append([line[5],line[6]])
    return guidepoints
