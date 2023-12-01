import os
import sys
from pathlib import Path

from bivme.preprocessing.cvi42.CVI42XML import *
from bivme.preprocessing.Contours import *


case = 'RV01'
output_path = Path(r'R:\resmed201900006-biomechanics-in-heart-disease\Sandboxes\Debbie\collaborations\chicago-rv-mesh\images\analysis\gpfiles-raw', f'{case}').as_posix()
dicom_extension = '.dcm' #.dcm
plot_contours = False

# path to dicom_metadata file obtained using extract_dicom_metadata.py
dcm_path = Path(r'R:\resmed201900006-biomechanics-in-heart-disease\Sandboxes\Debbie\collaborations\chicago-rv-mesh\images\chicago-cmr', f'{case}') #'/dicom_metadata.txt' 

#path to cvi42 file 
contour_file = Path(r'R:\resmed201900006-biomechanics-in-heart-disease\Sandboxes\Debbie\collaborations\chicago-rv-mesh\analysis\wsx', f'{case}.cvi42wsx').as_posix()

# do not change these output name files
out_contour_file_name = os.path.join(output_path,'GPFile.txt')
out_metadata_file_name = os.path.join(output_path,'SliceInfoFile.txt')

contour_name_map = { 'larvendocardialContour': 'LAX_RV_ENDOCARDIAL',
    				'larvepicardialContour': 'LAX_RV_EPICARDIAL ',
   					 'laendocardialContour': 'LAX_LV_ENDOCARDIAL',
   					'laepicardialContour':  'LAX_LV_EPICARDIAL' ,
    				"sarvendocardialContour": 'SAX_RV_ENDOCARDIAL',
    				 "sarvepicardialContour": 'SAX_RV_EPICARDIAL',
    				"saendocardialContour":'SAX_LV_ENDOCARDIAL',
    				"saepicardialContour": 'SAX_LV_EPICARDIAL',
					 'laxLaExtentPoints' : 'LAX_LA_EXTENT',
					 'laxRaExtentPoints' : 'LAX_RA_EXTENT',
					 'laxRvExtentPoints': 'LAX_RV_EXTENT',
					 'laxLvExtentPoints' : 'LAX_LV_EXTENT',
					 'laraContour': 'LAX_RA',
					 'lalaContour':'LAX_LA',
					 'saepicardialOpenContour':'SAX_LV_EPICARDIAL',
					 'saendocardialOpenContour': 'SAX_LV_ENDOCARDIAL',
					 'AorticValveDiameter':'AORTA_VALVE',
					 'PulmonaryValveDiameter':'PULMONARY_VALVE',
					 'AV':'AORTA_VALVE',
					 'MV':'MITRAL_VALVE',
                     "saepicardialContour": 'SAX_EPICARDIAL',
                     'apexEpiLv' : 'APEX_POINT'
					 }

cvi42Contour = CVI42XML(contour_file,dcm_path,dicom_extension,
                    convert_3D=True,log = True)

contour = cvi42Contour.contour
# add dict_of_frame in CVI42XML file , self.contour = Contours.contour ?
coords = contour.compute_3D_coordinates(timeframe=[])


if plot_contours == True:
    import matplotlib
    cmap = matplotlib.cm.get_cmap('gist_rainbow')

    contours_types = list(contour.points.keys())
    

    norm = matplotlib.colors.Normalize(vmin=1,vmax=2*len(contours_types))

    time_frame = [1]
    points_to_plot=[]

    for contour_type in contours_types:
        points_to_plot.append(
            cvi42Contour.contour.get_timeframe_points_coordinates(
            contour_type,time_frame)[1])


    cont_fig = visualization.Figure('contours')
    for index, points in enumerate(points_to_plot):
        cont_fig.plot_points(contours_types[index],points,
                             color=cmap(norm(2*index))[:3],size=1.5)


# write GPFile and SliceInfoFile
cvi42Contour.contour = contour
cvi42Contour.export_contour_points(out_contour_file_name)
cvi42Contour.export_dicom_metadata(out_metadata_file_name)

