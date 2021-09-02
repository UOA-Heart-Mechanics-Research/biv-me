Biventricular (RVLV) fitting codes

Original made by Charlène Mauger for her thesis at University of Auckland, NZ. These instructions will get a copy of the RVLV project up and running on your local machine for development and testing purposes.

# Prerequisites

## Python version and cvopt library

1. You need Python 3.5 to run it (I have not tried it on a later version). Using an earlier version of Python will give you errors as some of the syntax for numpy functions changed between Python 2 and Python 3. However, it can be run from a Python 3.5 virtual environment if you keep multiple Python versions. (see Running the script). 

2. You will need to change path line 5 in biventricularModel.py to link the python packages.

3. You also need to install the [cvxopt library](https://cvxopt.org/).

## Input

Two files are needed:

1. **GPfile.txt** which contains 3D contour points.

    Each point (row) is described as: 
        
    ```x y z label slice_number time_frame_number```

    * `x`, `y`, `z` are the 3D coordinates, 
    * `label` is the surface the point belongs to, 
    * `slice_number` is the dicom slice number it was extracted from, and 
    * `time_frame_number` is the time frame it was extracted from. 


