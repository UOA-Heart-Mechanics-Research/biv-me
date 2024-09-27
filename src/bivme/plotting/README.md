# Plotting guide points and meshes

Author: Laura Dal Toso, Charlene Mauger
Date: 26 May 2022

Usage
-----------------------------------------------

### Visualise GPFile
GPFiles can be visualise with plot_guidepoints located in src/bivme/ploting

usage: plot_guidepoints.py [-h] [-o OUTPUT_FOLDER] [-gp GP_DIRECTORY] [--gp_suffix GP_SUFFIX] [--si_suffix SI_SUFFIX]

This function plots a GPFile

options:
  -h, --help            show this help message and exit
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Path to the output folder
  -gp GP_DIRECTORY, --gp_directory GP_DIRECTORY
                        Define the directory containing guidepoint files
  --gp_suffix GP_SUFFIX
                        guidepoints to use if we do not want to fit all the models in the input folder
  --si_suffix SI_SUFFIX
                        Define slice info to use if multiple SliceInfo.txt file are available
