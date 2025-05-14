<div align="center">

# Biventricular modelling pipeline (biv-me)
![Python version](https://img.shields.io/badge/python-3.11-blue)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Test macOS** [![macOS](https://github.com/UOA-Heart-Mechanics-Research/biv-me/actions/workflows/macos.yml/badge.svg)](https://github.com/UOA-Heart-Mechanics-Research/biv-me/actions/workflows/macos.yml)

**Test Linux** [![Linux](https://github.com/UOA-Heart-Mechanics-Research/biv-me/actions/workflows/linux.yml/badge.svg)](https://github.com/UOA-Heart-Mechanics-Research/biv-me/actions/workflows/linux.yml)

**Test Windows** [![Windows](https://github.com/UOA-Heart-Mechanics-Research/biv-me/actions/workflows/windows.yml/badge.svg)](https://github.com/UOA-Heart-Mechanics-Research/biv-me/actions/workflows/windows.yml)

</div>

This repository provides an end-to-end pipeline for generating **guide point files (GPFiles)** from CMR DICOM data, fitting **biventricular models**, and computing **functional cardiac metrics** such as volumes, strains, and wall thickness.

Example data is available in the `example/` folder, including input DICOMs, output GPFiles, configuration files, and results for testing and reference.

For a detailed description of the end-to-end image to mesh pipeline, including image preprocessing and biventricular model fitting, please refer to:
**Dillon JR, Mauger C, Zhao D, Deng Y, Petersen SE, McCulloch AD, Young AA, Nash MP. An open-source end-to-end pipeline for generating 3D+t biventricular meshes from cardiac magnetic resonance imaging. In: Functional Imaging and Modeling of the Heart (FIMH) 2025. (in press)**

For a detailed description regarding the fitting of the biventricular model, please refer to:
**Mauger, C., Gilbert, K., Suinesiaputra, A., Pontre, B., Omens, J., McCulloch, A., & Young, A. (2018, July). An iterative diffeomorphic algorithm for registration of subdivision surfaces: application to congenital heart disease. In 2018 40th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC) (pp. 596-599). IEEE.** [DOI: 10.1109/EMBC.2018.8512394](https://doi.org/10.1109/EMBC.2018.8512394)

## 🚀 Installation Guide
-----------------------------------------------

The easiest way to set up this repository is to use the provided conda environment (python 3.11).
The conda environment named bivme311 can be created and activated by following steps 1-3 below.

### Step 1: Clone this repository
In a Git-enabled terminal, enter the following command to clone the repository.

```bash
git clone https://github.com/UOA-Heart-Mechanics-Research/biv-me.git
```
Alternatively, you can use software such as [GitHub Desktop](https://desktop.github.com/download/) or [GitKraken](https://www.gitkraken.com/) to clone the repository using the repository url.

### Step 2: Setup the virtual environment
If you have [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install) or [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main), you can create the conda virtual environment by entering the following commands into your terminal (or Anaconda Command Prompt if using Windows).

```bash
conda create -n bivme311 python=3.11
conda activate bivme311
```

### Step 3: Install the biv-me packages
Once the conda environment has been initialised, the necessary libraries and packages need to be installed. In your terminal, navigate to where you have cloned the repository to, e.g:

```bash
cd biv-me
```
Then, enter the following commands into your terminal to install the packages.

```bash
pip install -e .
python src/pyezzi/setup.py build_ext --inplace
```

If you do not already have them, guidepoint files (GPFiles) for personalised biventricular mesh fitting can be generated directly from CMR DICOM files. This requires installing additional packages, and downloading deep learning models for view prediction and segmentation. **If you do not plan to run preprocessing of CMR DICOM files to create GPFiles, you can skip the below steps.**

### (Optional) Step 4: Download deep learning models
The preprocessing code uses deep learning models for view prediction and segmentation. These models are located inside of a [different repository](https://github.com/UOA-Heart-Mechanics-Research/biv-me-dl-models), and are imported as a submodule. To download the models, you need to have Git LFS installed. You can run the following command in your terminal to check you have Git LFS.

```bash
git lfs install
```

If you don't have Git LFS installed, you can [follow these instructions to install it](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage). Once you have Git LFS installed, you can download the models by running the following command in your terminal. 

```bash
git submodule update --init
```
You can verify that the models have been downloaded by checking that the below directories contain .pth and .joblib files that are larger than 1 KB. If they don't, refer to the troubleshooting section below.

    src 
    └─── bivme
        └─── preprocessing
            └─── dicom
                └─── models
                    └─── Segmentation
                    └─── ViewSelection

### Troubleshooting
If this does not work for any reason, you can fall back on downloading the models manually. First, delete the folder called `src\bivme\preprocessing\dicom\models`. Create a new folder in its place, with the same name ('models'). Then, clone the repository with the deep learning models using the following command.

```bash
git clone https://github.com/UOA-Heart-Mechanics-Research/biv-me-dl-models.git
```

Copy the contents of the cloned models repository to `src\bivme\preprocessing\dicom\models` within your biv-me repository. Again, check if .pth and .joblib files are stored there, and that they are larger than 1 KB. If not, you probably do not have Git LFS initialised. Make sure to install Git LFS and retry. 

### (Optional) Step 5: Install additional libraries (PyTorch and nnU-Net)
This preprocessing code utilises PyTorch and nnU-Net. The default biv-me conda environment doesn't install either of these for you. To set these up, activate the biv-me conda environment by entering the following command into your terminal.

```bash
conda activate bivme311
```

Then, find the right PyTorch version for your GPU and OS and [install it as described on the website](https://pytorch.org/get-started/locally/).

**Do not install nnU-Net without installing PyTorch first!**. If you do, you will need to start over again.

After PyTorch has been installed, install nnU-Net by entering the following command into your terminal.

```bash
pip install nnunetv2
```

## Table of Contents
- [Installation](#🚀-installation-guide)

- [Generating biventricular models](#how-to-run-biv-me)
    - [Preprocessing DICOM data](#preprocessing)
    - [Fitting biventricular models](#fitting)
    - [Running end-to-end pipeline](#end-to-end-pipeline-preprocessing-and-fitting)
- [Analysis of models](#analysis-of-models)  
  - [Calculating volumes from models](#calculating-volumes-from-models)  
  - [Calculating strains from models](#calculating-strains-from-models)  
  - [Calculating wall thickness from models](#calculating-wall-thickness-from-models)
- [Postprocessing of models (experimental)](#postprocessing-of-models)
- [Contact us](#contact)    

-----------------------------------------------

## How to run biv-me
biv-me can be broadly divided into three different modules: **preprocessing** (DICOMs -> GPFiles), **fitting** (GPFiles -> biventricular models), and **analysis** (biventricular models -> metrics). 

The first two modules (preprocessing and/or fitting) can be run from `src/bivme/main.py`, as detailed below.

```bash
usage: main.py [-h] [-config CONFIG_FILE]
```

| **Argument**          | **Description**                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------- |
| `-h, --help`          | Displays the help message and exits.                                                          |
| `-config CONFIG_FILE`     | Path to config file describing which modules to run and their associated parameters.                      |

To run preprocessing and/or fitting, a **config file** must be created. The config file allows you to choose which modules to run and how you would like them to be run. An example of a config file can be found in `src/bivme/configs/config.toml`. If you wish, you can create a new config file for each time you want to make changes. Just make sure to update the path of the config file when you run the code!

#### **Example usage** <br>
Example DICOMs are provided in `example/dicoms` and example GPFiles are provided in `example/guidepoints/default`. You can verify that the repository is working by running biv-me on this example case (called 'patient1'), using the following commands.

```python
cd src/bivme
python main.py -config configs/config.toml
```
By default, this will generate GPFiles from the provided DICOMs for patient1 in `example/dicoms`, fit biv-me models to those GPFiles, and save them to the `src/output` directory. You can review the default paths by opening the config file at `src/bivme/configs/config.toml`.

**If you did not configure the preprocessing in Steps 4 and 5 of the installation, you will not be able to run preprocessing**. If so, make sure to set preprocessing=False in the config file before running. If you turn off preprocessing, running main.py will carry out fitting only on the example GPFiles in `example/guidepoints/default`.

Example fitted biv-me models for patient1 are provided in `example/fitted-models/default`. These are provided in .vtk, .obj, .txt, and .html versions. The .html is only stored for one frame due to storage considerations. 

### Preprocessing DICOM data
If you choose to run preprocessing, then **GPFiles** (GPFile_000.txt for frame 0, GPFile_001.txt for frame 1...) and one **SliceInfoFile.txt** will be created for each case for which there is DICOM data. These files are required for biventricular model fitting. GPFiles describe contour coordinates, whereas the SliceInfoFile.txt contains slice metadata.

When running preprocessing, you will need to provide certain directories in the config file. **All directories will be created upon runtime for you**, except for the 'source' directory. This should point to your DICOMs, which should be separated into folders by case like so:

    source
    └─── case1
        │─── *
    └─── case2
        │─── *
    └─── ...

As long as the DICOM images are organised separately by case, **they can be arranged in any way that you like**. There is no need to manually exclude non-cines prior to running, as the code should find which images are cines and which ones aren't by checking key terms within the series descriptions. Check `src/bivme/preprocessing/dicom/extract_cines.py` for the list of key terms, and update as needed for your dataset.

### Fitting biventricular models
If you choose to run fitting, then biventricular models will be created for each case for which there are GPFiles and a SliceInfoFile.txt file (if they already exist). Models will be generated as .txt files containing mesh vertices, html plots for visualisation, and (optionally) .obj or .vtk files for LV endocardial, RV endocardial, and epicardial meshes.

### End-to-end pipeline (preprocessing and fitting)
If you choose to run both preprocessing and fitting, they will run in sequence, such that biventricular models will be generated for each case for which there is DICOM data. 


## Analysis of models
Several tools are provided for the analysis of biventricular models, including scripts for volume calculation, strain analysis (circumferential and longitudinal strains), and wall thickness measurement. 

### Calculating volumes from models 
The script for calculating the volume of a mesh can be found in the `src/bivme/analysis` directory. It uses the tetrahedron method, which decomposes the mesh into tetrahedra and computes their volumes.

#### **Running the script** 
To run the `compute_volume.py` script, use the following command:

```bash
usage: compute_volume.py [-h] [-mdir MODEL_DIR] [-o OUTPUT_PATH] [-b BIV_MODEL_FOLDER] [-pat PATTERNS] [-p PRECISION]
```

| **Argument**          | **Description**                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------- |
| `-h, --help`          | Displays the help message and exits.                                                          |
| `-mdir MODEL_DIR`     | Specifies the path to the directory containing the biventricular models.                      |
| `-o OUTPUT_PATH`      | Specifies the directory where the output files will be saved.                                 |
| `-b BIV_MODEL_FOLDER` | Path to the folder containing the subdivision matrices for the models (default: `src/model`). |
| `-pat PATTERNS`       | The folder pattern to include for processing. You can use wildcards (default: `*`).           |
| `-p PRECISION`        | Sets the output precision (default: 2 decimal places).                                        |

#### **Example Usage** 
Example data is available in `example/fitted-models`. To compute the volumes using this data, run the following command:

```python
cd src/bivme/analysis
python compute_volume.py -mdir ../../../example/fitted-models -p 1 -o example_volumes
```

This will process the biventricular models in the `../../../example/fitted-models` directory, compute the volumes with a precision of 1 decimal place, and save the results in the `example_volumes` directory. The volumes will be saved in the `lvrv_volumes.csv` file.

**Sample Output** <br> 
The output file will look like this:

| **Name**     | **Frame** | **LV Volume (lv_vol)** | **LV Mass (lvm)** | **RV Volume (rv_vol)** | **RV Mass (rvm)** | **LV Epicardial Volume (lv_epivol)** | **RV Epicardial Volume (rv_epivol)** |
|--------------|-----------|------------------------|-------------------|------------------------|-------------------|---------------------------------------|--------------------------------------|
| patient_1    | 0         | 241.5                  | 218.3             | 220                    | 62.5              | 449.4                                 | 279.5                                |
| patient_1    | 1         | 252.2                  | 223.8             | 225.9                  | 69.3              | 465.4                                 | 291.8                                |


### Calculating strains from models 

The script for calculating both global circumferential and global longitudinal strains of a mesh can be found in the `src/bivme/analysis` directory. Geometric strain is defined as the change in geometric arc length from ED to any other frame using a set of predefined points and calculated using the Cauchy strain formula. The global circuferential strains are calculated at three levels: apical, mid and basal. The global longitudinal strains are calculated on a 4Ch and a 2Ch view.

#### **Running the scripts** 
To run the `compute_global_circumferential_strain.py` and `compute_global_longitudinal_strain.py` scripts, use the following command:

for circumferential strain:
```bash
usage: compute_global_circumferential_strain.py [-h] [-mdir MODEL_DIR] [-o OUTPUT_PATH] [-b BIV_MODEL_FOLDER] [-pat PATTERNS] [-p PRECISION] [-ed ED_FRAME]
 ```

for longitudinal strain:
```bash
usage: compute_global_longitudinal_strain.py [-h] [-mdir MODEL_DIR] [-o OUTPUT_PATH] [-b BIV_MODEL_FOLDER] [-pat PATTERNS] [-p PRECISION] [-ed ED_FRAME]
 ```

| **Argument**          | **Description**                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------- |
| `-h, --help`          | Displays the help message and exits.                                                          |
| `-mdir MODEL_DIR`     | Specifies the path to the directory containing the biventricular models.                      |
| `-o OUTPUT_PATH`      | Specifies the directory where the output files will be saved.                                 |
| `-b BIV_MODEL_FOLDER` | Path to the folder containing the subdivision matrices for the models (default: `src/model`). |
| `-pat PATTERNS`       | The folder pattern to include for processing. You can use wildcards (default: `*`).           |
| `-p PRECISION`        | Sets the output precision (default: 2 decimal places).                                        |
| `-ed ED_FRAME` | defines which frame is the ED frame. (default: 1st frame)


#### **Example Usage**
Example data is available in `example/fitted-models`. To compute the circuferential strains using this data, run the following command:

```python
cd src/bivme/analysis
python compute_global_circumferential_strain.py -mdir ../../../example/fitted-models -p 1 -o example_strains -ed 0
```

This will process the biventricular models in the `../../../example/fitted-models` directory, compute the global circumferential strain with a precision of 1 decimal place, and save the results in the `example_strains` directory. The GCS will be saved in the `global_circumferential_strain.csv.csv` file. The first frame will be used as ED. 

**Sample Output** <br>
The output file will look like this:

| **name**       | **frame** | **lv_gcs_apex** | **lv_gcs_mid** | **lv_gcs_base** | **rvfw_gcs_apex** | **rvfw_gcs_mid** | **rvfw_gcs_base** | **rvs_gcs_apex** | **rvs_gcs_mid** | **rvs_gcs_base** |
|------------|-------|-------------|------------|-------------|----------------|---------------|----------------|---------------|--------------|---------------|
| patient_1 | 0     | 0           | 0          | 0           | 0              | 0             | 0              | 0             | 0            | 0             |
| patient_1 | 1     | 0.019710907 | 0          | -0.00747012 | -0.011037528   | 0.005964215   | 0.034870641    | 0.038321168   | 0.032425422  | 0.0087241     |


### Calculating wall thickness from models <br>
The script computing the wall thickness can be found in src/bivme/analysis. Wall thickness is calculated on binary 3D images using [pyezzi](https://pypi.org/project/pyezzi/) for both LV and RV separately. The septal wall is included in the LV calculation and excluded from the RV. 

To run the `compute_wall_thickness.py` script, use the following command:

```bash
usage: compute_global_circumferential_strain.py [-h] [-mdir MODEL_DIR] [-o OUTPUT_PATH] [-b BIV_MODEL_FOLDER] [-pat PATTERNS] [-r VOXEL_RESOLUTION] [-s SAVE_SEGMENTATION_FLAG]
 ```

| **Argument**          | **Description**                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------- 
| `-h, --help`          | Displays the help message and exits.                                                          |
| `-mdir MODEL_DIR`     | Specifies the path to the directory containing the biventricular models.                      |
| `-o OUTPUT_PATH`      | Specifies the directory where the output files will be saved.                                 |
| `-b BIV_MODEL_FOLDER` | Path to the folder containing the subdivision matrices for the models (default: `src/model`). |
| `-pat PATTERNS`       | The folder pattern to include for processing. You can use wildcards (default: `*`).           |
| `-r VOXEL_RESOLUTION`        | Voxel resolution to compute the masks.                                        |
| `-s SAVE_SEGMENTATION_FLAG` | Boolean flag indicating whether to save 3D masks


#### **Example Usage** <br>
Example data is available in `example/fitted-models`. To compute the wall thickness using this data, run the following command:

```
python compute_wall_thickness.py -mdir ../../../example/fitted-models -o example_thickness
```

This will process the biventricular models in the `../../../example/fitted-models` directory, compute the wall thickness at a resolution of 1mm, and save the results in the `example_thickness` directory. Wall thickness is sampled and saved at the location of each vertex and can be visualised in Paraview as vertex color or in 3D slicer.

Adding the `-s` flag to the above command will also generate 4 extra nifti files per model: 2 3D masks with background=0, cavity=1, and wall=2 (`labeled_image_lv*.nii` and `labeled_image_lv*.nii`) and 2 3D mask containing thickness values at each voxel (`lv_thickness*.nii` and `rv_thickness*.nii`).

```
python compute_wall_thickness.py -mdir ../../../example/fitted-models -o example_thickness -s
```

**Sample Output** <br>
The output files will look like this in 3D Slicer:
![Wall_thickness](images/WallThickness.png)


## Postprocessing of models (experimental)
An experimental tool is available to refine the models by applying collision detection to prevent intersections between the RV septum and RV free wall. While the diffeomorphic fitting ensures no intersection between the endocardial and epicardial surfaces, this tool specifically addresses any potential self-intersections within the endocardial surface.

The script refitting a biv-me model with collision detection can be found in `src/bivme/postprocessing`. This script re-fit the models, using an extra collision detection step. An inital fit of the model is required as this will be used as guide points.

To run the detect_intersection.py script, use the following command:

```bash
usage: detect_intersection.py [-h] [-config CONFIG_FILE]
 ```

| **Argument**          | **Description**                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------- 
| `-h, --help`          | Displays the help message and exits.                                                          |
| `-mdir MODEL_DIR`     | Specifies the path to the directory containing the biventricular models.                      |
| `-o OUTPUT_PATH`      | Specifies the directory where the output files will be saved.                                 |
| `-b BIV_MODEL_FOLDER` | Path to the folder containing the subdivision matrices for the models (default: `src/model`). |
| `-pat PATTERNS`       | The folder pattern to include for processing. You can use wildcards (default: `*`).           |
| `-r VOXEL_RESOLUTION`        | Voxel resolution to compute the masks.                                        |
| `-s SAVE_SEGMENTATION_FLAG` | Boolean flag indicating whether to save 3D masks

The config file should be the one used to fit the original models. Refitted models will be saved in config["output_fitting"]["output_directory"]/corrected_models.

## Contribution - Notation
-----------------------------------------------
If you wish to contribute to this project, please follow the naming conventions outlined below:

| **Category**         | **Naming Convention**                                               | **Example**                                               |
|----------------------|---------------------------------------------------------------------|-----------------------------------------------------------|
| **Variable**         | Lowercase letters, words separated by underscores (snake_case)      | `site_name` instead of `sitename`                         |
| **Function/Method**  | Lowercase letters, words separated by underscores (snake_case)      | `def my_function()` instead of `def MyFunction()`          |
| **Constant**         | Uppercase letters, words separated by underscores                   | `MY_CONSTANT = 3.1416` instead of `MYCONSTANT = 3.1416`    |
| **Class**            | CamelCase                                                          | `class MyClass:` instead of `class myclass:`               |
| **Package/Module**   | No underscores or hyphens, consistent with Python standard library | `mypackage` instead of `my_package_name_with_underscores`  |
| **Type Variable**    | CamelCase with a leading capital letter                             | `Dict[int, str]` instead of `dict[int, str]`               |
| **Exception**        | Ends with “Error” suffix                                           | `class MyCustomExceptionError:` instead of `class MyCustomException:` |
| **Characters**       | Stick to ASCII characters                                          | `count = 42` instead of `ç = 42`                           |
| **Type Hints**       | Always use type hints for code readability                          | `def greet(name: str) -> str:` instead of `def greet(name):` |


## Acknowledgments
------------------------------------
This work is based on contributions by **Laura Dal Toso**, **Anna Mira**, **Liandong Lee**, **Richard Burns**, **Debbie Zhao**, **Joshua Dillon**, and **Charlène Mauger**.

## Contact
For questions or issues, please open an issue on GitHub or contact [joshua.dillon@auckland.ac.nz](joshua.dillon@auckland.ac.nz) or [charlene.1.mauger@kcl.ac.uk](charlene.1.mauger@kcl.ac.uk) 
