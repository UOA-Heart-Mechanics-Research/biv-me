# Plotting guide points and meshes

Author: Laura Dal Toso

Date: 26 May 2022

## Setup


Download the [BiV_Modelling](https://github.kcl.ac.uk/YoungLab/BiV_Modelling) repository.


## Contents


- Plot_html: to plot html files from the Model.txt files 
- Plot_GuidePoints: enambles the visualisation of guide points
- Split_Model_Files: splits a single .txt file containing mesh coordinates for all frames to many .txt files, one for each time frame. 
- vtk: This script saves the meshes and guide points in separate .vtk files.

## Usage

Use Plot_GuidePoints to visualise the guide points as html files. This is useful to quickly check if there are missing/wrong points in the input data. This is currently working on a single frame. 

Use Plot_html to visualise the models generated using the BIVfitting code as single html files or as time series. 

Use Split_Model_Files in case you have a single file containing the mesh coordinates for all frames. This step is necessary if the goal is to visualise time series using Paraview. Paraview will automatically acquire separate .txt files as a time series if consecutive frame numbers are included in the file name.

Use vtk to save the models and guide points in vtk files. 
