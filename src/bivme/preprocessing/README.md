Preprocessing Module
-----------------------------------------------
Author: Laura Dal Toso

The scripts in this repository can be used to preprocess GPFiles and SliceInfo files. 

## Setup

To use these scripts, it is necessary to download the [BiV_Modelling](https://github.kcl.ac.uk/YoungLab/BiV_Modelling) repository.



## Contents

- mesh tools: tools for operations on mesh objects
- results : folder where the outputs are stored
- test : contains a test dataset


- Contours.py: Class to read/write/modify GPFiles
- CVI42XML: This class reads an xml file from CVI42 and extracts the 2D points 
- main_preprocessing.py: main file that does the pre-processing 
- parallel_preprocessing.py: This file calls main_preprocessing.py, and performs the preprocessing on multiple CPUs in parallel
- utils.py: contains functions for the pre-processing


## Usage

The first step is to copy your input data in the test folder. Each subfolder of 'test' should contain one GPFile and one SliceInfoFile relative to the same patient.

Main_preprocessing.py performs the pre processing sequentially on a list of folders. Once you open this script, you can focus on the lines after if __name__ == '__main__'. 

**Step1: Change relative paths**

Change relative paths and check that the variables initial_gpfile and initial_sliceinfo match the file names that you have as inputs in the test subfolders

**Step2: Select preprocessing operations**

This script allows to perform three main preprocessing operations, which can be tuned on/off setting to True/False these three parameters: do_landmarks_tracking, clean_contours, find_EDES_frames. The function of these parameters are hereby listed: 

- do_landmarks_tracking controls the landmark tracking. This can be done if one wats to know how many given landmarks, which are mainly the valve points and apex, are present in the GPFile at each time frame. The landmark tracking can also calculate missing landmarks, if they are missing from max 2 conecutive time frames. If landmarks are missing from more than 2 frames teh case is discarded. 

- clean_contours controls the deletion of unwanted points from the input contours. This can be used to delete selected contours (i.e. 3 chamber LAX_LV_EPICARDIUM) or a set of points. Currently, the script deletes the points located in LAX contours and between two valve points. We noticed that these points were deforming the final models.

- find_EDES_frames finds where the ED and ES frames are located, in a time series, based on the distance between mitral valve points and apex. It outputs a spreadsheet with patient names and ED/ES frame numbers

**Step3: Perform pre-processing**

Once these variables are set, you can run the script typing 
$python Main_preprocessing.py
to get the processed GPFiles and SliceInfoFiles which are ready for fitting.

**Processing in parallel**

There is the possibility to run the pre-processing in parallel CPUs. To do so, use parallel_preprocessing.py insted of main_preprocessing.py. You can set the number of workers, which is the number of CPUs to use. Then set the same parameters described in Step2, and run the script. 


## Credits

This work is based on code developed by Charlene Mauger, Anna Mira and Richard Burns 
