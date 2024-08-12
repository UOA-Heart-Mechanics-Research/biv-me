# Preprocessing

These scripts can be used to generate GPFiles and SliceInfoFiles from CVI42 contour files (in .wsx format) required for BiV fitting.

## Contents

- cvi42_to_gp: main file for the conversion from cvi42 files to GPFiles
- CVI42XML: This class reads an xml file from CVI42 and extracts the 2D points
- parse_cvi42_xml: functions to parse xml file and save its content

## Usage

Run the script cvi42_to_gp.py. The user will need to change the output_path, the dcm_path and the contour_file path. The outputs will be a GPFile.txt and a SliceInfoFile.txt.

## Credits

Laura Dal Toso, Anna Mira, Richard Burns, Charlene Mauger
