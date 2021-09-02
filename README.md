# Biventricular (RVLV) fitting codes

Original made by Charlène Mauger for her thesis at University of Auckland, NZ. These instructions will get a copy of the RVLV project up and running on your local machine for development and testing purposes.

## Prerequisites

### Python version and cvopt library

1. You need Python 3.5 to run it (I have not tried it on a later version). Using an earlier version of Python will give you errors as some of the syntax for numpy functions changed between Python 2 and Python 3. However, it can be run from a Python 3.5 virtual environment if you keep multiple Python versions. (see Running the script). 

2. You will need to change path line 5 in biventricularModel.py to link the python packages.

3. You also need to install the [cvxopt library](https://cvxopt.org/).

### Input

Two files are needed:

1. **GPfile.txt** which contains 3D contour points.

    Each point (row) is described as: 
        
    ```x y z label slice_number time_frame_number```

    * `x`, `y`, `z` are the 3D coordinates, 
    * `label` is the surface the point belongs to, 
    * `slice_number` is the dicom slice number it was extracted from, and 
    * `time_frame_number` is the time frame it was extracted from. 

2. ***SliceInfoFile.txt** which contains dicom info.

For the surface label, these are provided by the dataset:

* `RVFW`: RV free wall 
* `RVS`: RV septum 
* `saendocardialContour`: LV endocardium (extracted from a short axis slice)
* `saepicardialContour`: LV epicardium (extracted from a short axis slice)
* `BP_points`: mitral points 
* `Triscipid_Valve`: tricuspid points 
* `Aorta`: aorta points 
* `Pulmonary`: pulmonary points 
* `laraContour`: RA contour (extracted from a long axis slice)
* `lalaContour`: LA contour (extracted from a long axis slice)
* `RV_insert`: RV insertion points (where RV septum and RV free-wall meet)
* `LA_Apex_Point`: apex
* `laendocardialContour`: LV endocardium (extracted from a long axis slice)
* `laepicardialContour`: LV epicardium (extracted from a long axis slice)
* `larvendocardialContour`: RV endocardium (extracted from a long axis slice)

You can also add new labels to adapt to your dataset.

## Running the script

Two scripts are provided to make sure the code is working for you.

1. **example_step_by_step.ipynb**: this script shows you how the code works step by step using Jupyter Notebook and the different steps to follow when fitting one model. 

    You need to install [jupyter notebook](https://jupyter.org/) and then run, open and execute each cell one-by-one.

2. **example.py**: this script shows you how to use the code when you want to fit more than one time frame (the first 3 time frames are fitted in this example). To run it, open a command-line or terminal and enter (depending on your Python installation)

## Author

Charlène Mauger, University of Auckand, c.mauger@auckland.ac.nz or charlene.mauger1@gmail.com