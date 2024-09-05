
Biventricular model fitting framework

-----------------------------------------------
This is an import of the KCL BiV modelling code (originally called BiV_Modelling).

Date: 01 December 2023

-----------------------------------------------

This code performs patient-specific biventricular mesh customization. 

The process takes place in 2 steps:
1. correction of breath-hold misregistration between short-axis slices, and 
2. deformation of the subdivided template mesh to fit the manual contours while preserving 
the topology of the mesh.

Documentation: https://github.kcl.ac.uk/pages/YoungLab/BiV_Modelling/


Contents 
-----------------------------------------------
- BiVFitting: contains the code that performs patient-specific biventricular mesh customization. 
- model: contains .txt files required by the fitting modules
- results: output folder
- test_data: contains one subfolder for each patient. Each subfolder contains the GPFile and SliceInfoFile relative to one patient.
- config_params: configuration of parameters for the fitting
- perform_fit: script that contains the routine to perform the biventricular fitting
- run_parallel: allows fitting using parallel CPUs. Each patient is assigned to one CPU.

Installation ![Python versions](https://img.shields.io/badge/python-3.11-blue)
-----------------------------------------------
The easiest way to get this repo setup is to use the provided conda environment (python 3.11).
The conda environment named biv311 can be created and activated with

```
cd biv-me
conda create -n bivme311 python=3.11
conda activate bivme311
```

Install the bivme package
```
pip install -e .
```

Notation
-----------------------------------------------
If you wish to contribute to this project, we ask you to follow the naming conventions below :
- **Variable**: use lowercase word. A variable that use multiple words should be separated with an underscore (snake case)
```sitename``` should be written as ```site_name```
- **Function and Method**: function/method names should follow the PEP 8 naming conventions ```def MyFunction()``` should be written as ```def my_function()```
- **Constant**: constant names should be written in uppercase letters with underscores separating words ```MYCONSTANT = 3.1416``` should be written ```MY_CONSTANT = 3.1416```
- **Class**; class names should follow the CamelCase convention: ```class myclass:``` should be written as ```class MyClass:```
- **Package and Module** : Avoid using underscores or hyphens in package names to maintain consistency with the Python standard library and third-party packages. ```my_package_name_with_underscores = ...``` should be written ```mypackage = ...```
- **Type variable**: follow the convention of using CamelCase with a leading capital letter: ```def my_function(items: dict[int, str]):``` should be written as ```def my_function(items: Dict[int, str]):```
- **Exception** exception names should have the suffix “Error.”: ```class MyCustomException:``` should be ```class MyCustomExceptionError:```
- Stick to ASCII characters to ensure smooth collaboration and consistent code execution: avoid for example ```ç = 42```. Instead prefer ```count = 42```
- Use type hints for code readability and prevent type-related errors.
```
def greet(name):
    return "Hello, " + name
```
should be
```
def greet(name: str) -> str:
    return "Hello, " + name
```

Usage
-----------------------------------------------

### Fit a Biv-me model to GP files
The script for the mesh fitting can be found in src/bivme/fitting
```
usage: perform_fit.py [-h] [-config CONFIG_FILE]

Biv-me

options:
  -h, --help            show this help message and exit
  -config CONFIG_FILE, --config_file CONFIG_FILE
                        Config file containing fitting parameters

```

### Calculate volumes from biv-me models
The script for the volume calculation can be found in src/bivme/analysis

```
usage: compute_volume.py [-h] [-mdir MODEL_DIR] [-o OUTPUT_FILE] [-b BIV_MODEL_FOLDER] [-pat PATTERNS] [-p PRECISION]

  -h, --help            show this help message and exit
  -mdir MODEL_DIR, --model_dir MODEL_DIR
                        path to biv models
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        output path
  -b BIV_MODEL_FOLDER, --biv_model_folder BIV_MODEL_FOLDER
                        folder containing subdivision matrices
  -pat PATTERNS, --patterns PATTERNS
                        folder patterns to include (default '*')
  -p PRECISION, --precision PRECISION
                        Output precision (default: 2)
```
Results will be saved in {OUTPUT_PATH}/lvrv_volumes.csv

### Calculate strains from biv-me models
The script for strain calculation can be found in src/bivme/analysis. Geometric strain is defined as the change in geometric arc length from ED to ES using a set of predefined points and calculated using the Cauchy strain formula.

**For global circumferential strain calculation**
```
usage: compute_global_circumferential_strain.py [-h] [-mdir MODEL_DIR] [-o OUTPUT_PATH] [-b BIV_MODEL_FOLDER] [-pat PATTERNS] [-ed ED_FRAME]
                                                [-p PRECISION]

Global circumferential strain calculation

options:
  -h, --help            show this help message and exit
  -mdir MODEL_DIR, --model_dir MODEL_DIR
                        path to biv models
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        output path
  -b BIV_MODEL_FOLDER, --biv_model_folder BIV_MODEL_FOLDER
                        folder containing subdivision matrices
  -pat PATTERNS, --patterns PATTERNS
                        folder patterns to include (default '*')
  -ed ED_FRAME, --ed_frame ED_FRAME
                        ED frame
  -p PRECISION, --precision PRECISION
                        Output precision


```
Results will be saved in {OUTPUT_PATH}/global_circumferential_strain.csv

**For global longitudinal strain calculation**
```
usage: compute_global_longitudinal_strain.py [-h] [-mdir MODEL_DIR] [-o OUTPUT_PATH] [-b BIV_MODEL_FOLDER] [-pat PATTERNS] [-ed ED_FRAME]
                                             [-p PRECISION]

Global longitudinal strain calculation

options:
  -h, --help            show this help message and exit
  -mdir MODEL_DIR, --model_dir MODEL_DIR
                        path to biv models
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        output path
  -b BIV_MODEL_FOLDER, --biv_model_folder BIV_MODEL_FOLDER
                        folder containing subdivision matrices
  -pat PATTERNS, --patterns PATTERNS
                        folder patterns to include (default '*')
  -ed ED_FRAME, --ed_frame ED_FRAME
                        ED frame
  -p PRECISION, --precision PRECISION
                        Output precision

```
Results will be saved in {OUTPUT_PATH}/global_longitudinal_strain.csv

### Compute wall thickness
The script for computing the wall thickness can be found in src/bivme/analysis. Wall thickness is calculated on binary 3D images using pyezzi (https://pypi.org/project/pyezzi/) for both LV and RV separately. The septal wall is included in the LV calculation and excluded from the RV 

```
usage: compute_wall_thickness.py [-h] [-mdir MODEL_DIR] [-o OUTPUT_FOLDER] [-b BIV_MODEL_FOLDER] [-pat PATTERNS] [-r VOXEL_RESOLUTION] [-s]

Wall thickness calculation from 3D masks

options:
  -h, --help            show this help message and exit
  -mdir MODEL_DIR, --model_dir MODEL_DIR
                        path to biv models
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        output path
  -b BIV_MODEL_FOLDER, --biv_model_folder BIV_MODEL_FOLDER
                        folder containing subdivision matrices
  -pat PATTERNS, --patterns PATTERNS
                        folder patterns to include (default '*')
  -r VOXEL_RESOLUTION, --voxel_resolution VOXEL_RESOLUTION
                        Output precision
  -s, --save_segmentation
                        Boolean value indicating if we want the 3D masks to be saved


```

Results and segmentation masks will be saved in {OUTPUT_PATH}/wall_thickness. Wall thickness is saved at each vertex in the mesh

**Step 4**

After changing the script perform_fit.py according to your needs, there are two options to perform the model fitting. The first option is to fit the list of inout patients sequentially, by running perform_fit.py. 

To speed up the fitting, you may want to process different cases in parallel CPUs. To perform the fitting in parallel, you can use run_parallel.py. The relative paths at the bottom of this script need to be changed first, then the script can be lauched to generate the fitted models.

Credits
------------------------------------
Based on work by: Laura Dal Toso, Anna Mira, Liandong Lee, Richard Burns, Charlene Mauger
