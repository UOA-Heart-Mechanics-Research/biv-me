# DICOM_processing
Author: Laura Dal Toso

These scripts can be used to generate SliceInfoFiles and GPFiles, starting respectively from 2D contours saved in cvi42 format and DICOM images.

## Setup

To use these scripts, it is necessary to download the [BiV_Modelling](https://github.kcl.ac.uk/YoungLab/BiV_Modelling) repository.

## Contents

- Contours: Class to read/write/modify GPFiles
- cvi42_to_gp: main file for the conversion from cvi42 files to GPFiles
- CVI42XML: This class reads an xml file from CVI42 and extracts the 2D points
- extract_dicom_metadata: main file for the extraction of metadata from DICOM
- parse_cvi42_xml: functions to parse xml file and save its content

## Usage


**STEP1: Extraction of metadata from DICOM**

The first step is usually to extract the metadata from the DICOM images. This can be done using the file extract_dicom_metadata.py. At first the user will need to change the dicom_dir path and the output path. Then the user can run the script, the outpur will be a file called 'dicom_metadata'.txt.

**STEP2: Creation of GPFIle and SliceInfoFile**

The second step is to generate the GPFile and SliceInfoFile. To do so, use the script cvi42_to_gp. Change the output_path, the dcm_path and the contour_file path. You ca then set plot_contours to True/False to get a plot of the contours. After thsi you can run the file, the outputs will be a SliceInfoFile and a GPFile



## Credits

Anna Mira, Richard Burns, Charlene Mauger

