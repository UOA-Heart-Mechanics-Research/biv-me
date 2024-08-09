import os
import sys
from pathlib import Path

from bivme.preprocessing.cvi42.CVI42XML import *
from bivme.preprocessing.Contours import *


if __name__ == "__main__":
    """
    This script converts CVI42 .wsx files to guidepoint (GP) files for BiV fitting.
    """

    # set list of cases to process
    caselist = ["SCMR_2_corrected", "SCMR_3", "SCMR_4", "SCMR_5"]
    
    # set directory of CVI42 .wsx files
    dir_cvi42 = r"R:\resmed201900006-biomechanics-in-heart-disease\Sandboxes\Josh\bivme\test\wsx"

    # set directory to save GP files
    dir_gp = r"R:\resmed201900006-biomechanics-in-heart-disease\Sandboxes\Josh\bivme\test\gpfiles"

    # set directory of corresponding DICOM images
    dir_dicom = r"R:\resmed201900006-biomechanics-in-heart-disease\Sandboxes\Josh\bivme\test\images"
    dicom_extension = ".dcm"

	# plot contours?
    plot_contours = False
    
    # --------------------
    # Begin processing...
    # --------------------

    for case in caselist:
        
        print(f"# Processing case {case}...")
        
        # path to cvi42 file
        contour_file = Path(dir_cvi42, f"{case}.cvi42wsx").as_posix()
        
        # path to dicom images
        dcm_path = Path(dir_dicom, f"{case}")

        # do not change these output name files
        output_path = Path(dir_gp, f"{case}").as_posix()
        out_contour_file_name = Path(output_path, "GPFile.txt").as_posix()
        out_metadata_file_name = Path(output_path, "SliceInfoFile.txt").as_posix()

        contour_name_map = {
            "larvendocardialContour": "LAX_RV_ENDOCARDIAL",
            "larvepicardialContour": "LAX_RV_EPICARDIAL ",
            "laendocardialContour": "LAX_LV_ENDOCARDIAL",
            "laepicardialContour": "LAX_LV_EPICARDIAL",
            "sarvendocardialContour": "SAX_RV_ENDOCARDIAL",
            "sarvepicardialContour": "SAX_RV_EPICARDIAL",
            "saendocardialContour": "SAX_LV_ENDOCARDIAL",
            "saepicardialContour": "SAX_LV_EPICARDIAL",
            "laxLaExtentPoints": "LAX_LA_EXTENT",
            "laxRaExtentPoints": "LAX_RA_EXTENT",
            "laxRvExtentPoints": "LAX_RV_EXTENT",
            "laxLvExtentPoints": "LAX_LV_EXTENT",
            "laraContour": "LAX_RA",
            "lalaContour": "LAX_LA",
            "saepicardialOpenContour": "SAX_LV_EPICARDIAL",
            "saendocardialOpenContour": "SAX_LV_ENDOCARDIAL",
            "AorticValveDiameter": "AORTA_VALVE",
            "PulmonaryValveDiameter": "PULMONARY_VALVE",
            "AV": "AORTA_VALVE",
            "MV": "MITRAL_VALVE",
            "saepicardialContour": "SAX_EPICARDIAL",
            "apexEpiLv": "APEX_POINT",
        }

        cvi42Contour = CVI42XML(
            contour_file, dcm_path, dicom_extension, convert_3D=True, log=True
        )

        contour = cvi42Contour.contour
        # add dict_of_frame in CVI42XML file , self.contour = Contours.contour ?
        coords = contour.compute_3D_coordinates(timeframe=[])

        if plot_contours:
            
            import matplotlib

            cmap = matplotlib.cm.get_cmap("gist_rainbow")

            contours_types = list(contour.points.keys())

            norm = matplotlib.colors.Normalize(vmin=1, vmax=2 * len(contours_types))

            time_frame = [1]
            points_to_plot = []

            for contour_type in contours_types:
                points_to_plot.append(
                    cvi42Contour.contour.get_timeframe_points_coordinates(
                        contour_type, time_frame
                    )[1]
                )

            cont_fig = visualization.Figure("contours")
            for index, points in enumerate(points_to_plot):
                cont_fig.plot_points(
                    contours_types[index],
                    points,
                    color=cmap(norm(2 * index))[:3],
                    size=1.5,
                )

        # write GPFile and SliceInfoFile
        cvi42Contour.contour = contour
        cvi42Contour.export_contour_points(out_contour_file_name)
        cvi42Contour.export_dicom_metadata(out_metadata_file_name)
print('Done')