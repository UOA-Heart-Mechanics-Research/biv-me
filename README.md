<div align="center">

### 12th May, 2026: [v1.2 major update - improved segmentation and optimised model fitting](https://github.com/UOA-Heart-Mechanics-Research/biv-me/releases/tag/v1.2)
Major update provides new segmentation and view selection models designed for greater generalisability, and significantly reworks model fitting for enhanced performance. 

### 9th September, 2025: [v1.1.6 update - new visualisation tools](https://github.com/UOA-Heart-Mechanics-Research/biv-me/releases/tag/v1.1.6)
Update adds tools to visualise images alongisde models to help visual inspection of mesh reconstruction and create eye-catching videos. 

### 28th July, 2025: [v1.1.5 update - improved view selection & smoother landmarks](https://github.com/UOA-Heart-Mechanics-Research/biv-me/releases/tag/v1.1.5)
Update significantly improves view selection by overhauling metadata-based prediction, and adds new config option for more temporally consistent landmarks.



# Biventricular modelling pipeline (biv-me) - **v1.2**
![Python version](https://img.shields.io/badge/python-3.11-blue)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Test macOS** [![macOS](https://github.com/UOA-Heart-Mechanics-Research/biv-me/actions/workflows/macos.yml/badge.svg)](https://github.com/UOA-Heart-Mechanics-Research/biv-me/actions/workflows/macos.yml)

**Test Linux** [![Linux](https://github.com/UOA-Heart-Mechanics-Research/biv-me/actions/workflows/linux.yml/badge.svg)](https://github.com/UOA-Heart-Mechanics-Research/biv-me/actions/workflows/linux.yml)

**Test Windows** [![Windows](https://github.com/UOA-Heart-Mechanics-Research/biv-me/actions/workflows/windows.yml/badge.svg)](https://github.com/UOA-Heart-Mechanics-Research/biv-me/actions/workflows/windows.yml)

</div>

This repository provides an end-to-end pipeline for generating guidepoint files (**GPFiles**) from CMR DICOMs, fitting biventricular models (**biv-me models**), and computing **functional cardiac metrics** such as volumes, strains, and wall thickness.

Example data is available in the `demo/` folder, including input DICOMs, GPFiles, and fitted models for testing and reference.

For a detailed description of the end-to-end image to mesh pipeline (v1.2.x), including image preprocessing and biventricular model fitting, please refer to: <blockquote> Dillon, J., Mauger, C., Zhao, D., Petersen, S. E., McCulloch, A. D., Young, A. A., & Nash, M. P. (2026). biv-me: Open-source software for generating time-varying biventricular meshes from cine cardiovascular magnetic resonance imaging with multi-cohort validation. Medical Image Analysis, 114, 104252. [DOI: 10.1016/j.media.2026.104252](https://doi.org/10.1016/j.media.2026.104252) </blockquote>

For a detailed description regarding the fitting of the biventricular model, please refer to: <blockquote>Mauger, C., Gilbert, K., Suinesiaputra, A., Pontre, B., Omens, J., McCulloch, A., & Young, A. (2018, July). An iterative diffeomorphic algorithm for registration of subdivision surfaces: application to congenital heart disease. In 2018 40th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC) (pp. 596-599). IEEE. [DOI: 10.1109/EMBC.2018.8512394](https://doi.org/10.1109/EMBC.2018.8512394)</blockquote>

Depending on how you use biv-me, please cite the relevant publication(s) above.  

CMR DICOMs           | Contours                   |        biv-me models                |
:-------------------------:|:-------------------------:|:-------------------------:|
![Images](images/image.gif) | ![contours](images/contours.gif) |![meshes](images/meshes.gif)

## 🚀 Installation Guide
-----------------------------------------------

The easiest way to set up this repository is to use the provided conda environment (python 3.11).
The conda environment named *bivme311* can be created and activated by following steps 1-3 below.

### Step 1: Clone this repository
In a Git-enabled terminal, enter the following command to clone the repository.

```bash
git clone https://github.com/UOA-Heart-Mechanics-Research/biv-me.git
```
Alternatively, you can use software such as [GitHub Desktop](https://desktop.github.com/download/) or [GitKraken](https://www.gitkraken.com/) to clone the repository using the repository url. If you are prompted to initialise submodules within these applications after cloning the repository, **select no**. We will initialise these later.

### Step 2: Setup the virtual environment
If you have [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install) or [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main), you can create the conda virtual environment by entering the following commands into your terminal (or Anaconda Command Prompt if using Windows).

```bash
conda create -n bivme311 python=3.11
conda activate bivme311
```

### Step 3: Install the biv-me packages
Once the conda environment has been initialised, the necessary libraries and packages need to be installed. In your terminal, navigate to where you have cloned the repository to, e.g.,

```bash
cd biv-me
```
Then, enter the following commands into your terminal to install the packages.

```bash
pip install -e .
python src/pyezzi/setup.py build_ext --inplace
```

If you do not already have them, guidepoint files (GPFiles) for biventricular model fitting can be generated directly from CMR DICOM files. This requires installing additional packages, and downloading deep learning models for view prediction and segmentation. **If you do not plan to run preprocessing of DICOM files to create GPFiles, you can skip Steps 4 and 5.**

### (Optional) Step 4: Download deep learning models
The preprocessing code uses deep learning models for view prediction and segmentation. These models are located inside of a [different repository](https://github.com/UOA-Heart-Mechanics-Research/biv-me-dl-models), and are imported as a submodule. To download the models, you need to have Git LFS installed. You can run the following command in your terminal to check you have Git LFS.

```bash
git lfs install
```

If you don't have Git LFS installed, you can [follow these instructions to install it](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage). Once you have Git LFS installed, you can download the models by running the following command in your terminal. 

```bash
git submodule update --init
```
By default, this will install the submodule associated with this version of biv-me. Refer to the [FAQs](#faqs) for more information on version control of deep learning models.

You can verify that the models have been downloaded by checking that the below directories contain .pth files that are larger than 1 KB. If they don't, refer to the troubleshooting section below.

    src 
    └─── bivme
        └─── preprocessing
            └─── dicom
                └─── models
                    └─── Segmentation
                    └─── ViewSelection

### Troubleshooting
If importing the submodule does not work for any reason, you can fall back on downloading the models manually. First, delete the folder called `src\bivme\preprocessing\dicom\models`. Create a new folder in its place, with the same name ('models'). Then, clone the repository with the deep learning models using the following command.

```bash
git clone https://github.com/UOA-Heart-Mechanics-Research/biv-me-dl-models.git
```

Copy the contents of the cloned models repository to `src\bivme\preprocessing\dicom\models` within your biv-me repository. Again, check if .pth and .joblib files are stored there, and that they are larger than 1 KB. If not, you probably do not have Git LFS initialised. Make sure to install Git LFS and retry. 

### (Optional) Step 5: Install additional libraries (PyTorch and nnU-Net)
This preprocessing code utilises PyTorch and nnU-Net. The default biv-me conda environment doesn't install either of these for you. To set these up, activate the biv-me conda environment by entering the following command into your terminal.

```bash
conda activate bivme311
```

Then, find the right PyTorch version for your GPU and OS and [**install it as described on the website**](https://pytorch.org/get-started/locally/).

### **Do not install nnU-Net without installing PyTorch first!** 
#### If you do, you will need to start over again. 

After PyTorch has been installed, install nnU-Net by entering the following command into your terminal.

```bash
pip install nnunetv2
```

## ⭐ Future updates 
We update biv-me regularly to apply patches and add new features. Make sure to star and/or watch the repository to keep up with new releases!

## Table of Contents
- [**Installation guide**](#🚀-installation-guide)

- [**How to run biv-me**](#how-to-run-biv-me)
    - [Example usage](#example-usage)
    - [Preprocessing DICOM data](#preprocessing-dicom-data)
    - [Fitting biv-me models](#fitting-biv-me-models)
    - [Running end-to-end pipeline](#running-end-to-end-pipeline-preprocessing-and-fitting)
- [**Visualisation of models (and images)**](#visualisation-of-models-and-images)
  - [Example .html plot (with images)](#example-html-plot-with-images)
  - [Example .vtk image and model sequence](#example-vtk-image-and-model-sequence) 
- [**Analysis of models**](#analysis-of-models)  
  - [Calculating volumes from models](#calculating-volumes-from-models)  
  - [Calculating strains from models](#calculating-strains-from-models)  
  - [Calculating wall thickness from models](#calculating-wall-thickness-from-models)
- [**Postprocessing of models (experimental)**](#postprocessing-of-models-experimental)
- [**FAQs**](#faqs)
- [**Contribution - notation**](#contribution---notation)
- [**Acknowledgements**](#acknowledgments)
- [**Contact us**](#contact)    

-----------------------------------------------

## How to run biv-me
biv-me can be broadly divided into three different modules: **preprocessing** (DICOMs -> GPFiles), **fitting** (GPFiles -> biv-me models), and **analysis** (biv-me models -> metrics). 

The first two modules (preprocessing and/or fitting) can be run from `src/bivme/main.py`, as detailed below.

```bash
usage: main.py [-h] [-config CONFIG_FILE] [-case CASE_NAME]
```

| **Argument**          | **Description**                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------- |
| `-h, --help`          | Displays the help message and exits.                                                          |
| `-config CONFIG_FILE`     | Path to config file describing which modules to run and their associated parameters.                      |
| `-case CASE_NAME`   | Name of individual case folder to run in configured directory (runs all if not provided).

To run preprocessing and/or fitting, a **config file** must be created. The config file allows you to choose which modules to run and how you would like them to be run. An example of a config file can be found in `src/bivme/configs/config.toml`. If you wish, you can create a new config file for each time you want to make changes. Just make sure to update the path of the config file when you run the code!

### Running the pipeline interactively
We also provide an option to run the pipeline in an interactive step-by-step manner. This pipeline is the same as the one in `main.py`, but it is structured as a Jupyter notebook instead. It is a great place to start if you want to troubleshoot or to learn how the code works. Open the notebook at `src/bivme/main_interactive.ipynb` to get the interactive version of biv-me started.

### Example usage 
Example DICOMs are provided in `demo/dicoms` and example GPFiles are provided in `demo/guidepoints/example`. You can verify that the repository is working by running biv-me on this demo case (called *patient1*), using the following commands.

```python
cd src/bivme
python main.py -config configs/config.toml
```

By default, this will generate new GPFiles from the DICOMs for *patient1* in `demo/dicoms` (**preprocessing**), fit biv-me models to the new GPFiles created in `demo/guidepoints/test-run` (**fitting**), and save the fitted models to the `demo/fitted-models/test-run` directory. You can review the default paths by opening the config file at `src/bivme/configs/config.toml`.

During preprocessing, you will be presented with a GUI that displays the automatically predicted views and prompts you to make corrections (if needed), save those corrections, and then continue (by closing the GUI). In this demo case, the automatically predicted views are correct, but they may not always be. To minimise downstream errors, it is recommended to run biv-me with the *correct_mode* set to 'manual' in the config file, as it is for the demo case. However, it is also possible to run fully automatically by setting *correct_mode* to 'automatic'.

**If you did not configure the preprocessing in Steps 4 and 5 of the installation, you will not be able to run preprocessing**. If so, make sure to set *preprocessing=False* in the config file before running. If you turn off preprocessing, running `src/bivme/main.py` will carry out fitting only on the already provided example GPFiles in `demo/guidepoints/example`.

#### **Interactive example**
You can also run the Jupyter notebook at `src/bivme/main_interactive.ipynb` to run the same demo case, which offers a chance to get more familiar with how the pipeline works. 

#### **Sample output**
Example biv-me models for *patient1* have been already fitted, and can be found in `demo/fitted-models/example`. These are provided in .txt, .vtk, and .obj formats. The first frame of the fitted models in .vtk format is visualised below using [Paraview](https://www.paraview.org/). Your fitted models should ideally look something like this.

![Model4ch](images/Model1.png) 

![Modelsax](images/Model2.png)

### Preprocessing DICOM data
When you run preprocessing, **GPFiles** (GPFile_000.txt for frame 0, GPFile_001.txt for frame 1...) and one **SliceInfoFile.txt** will be created for each case for which there is DICOM data. These files are required for biventricular model fitting. GPFiles describe contour coordinates, whereas the SliceInfoFile.txt contains slice metadata.

When running preprocessing on your own data, you will need to provide certain directories in the config file. **All directories will be created upon runtime for you**, except for the 'source' directory. This should point to your DICOMs, which should be separated into folders by case like so:

    source
    └─── case1
        │─── *
    └─── case2
        │─── *
    └─── ...

As long as the DICOM images are organised separately by case, **they can be arranged in any way that you like**. There is no need to manually exclude non-cines prior to running, as the code should find which images are cines and which ones aren't by checking key terms within the series descriptions. Check `src/bivme/preprocessing/dicom/extract_cines.py` for the list of key terms, and update as needed for your dataset.

### Fitting biv-me models
When you run fitting, biv-me models will be created for each case for which there are GPFiles and a SliceInfoFile.txt file. 

If you already have GPFiles, then you do not need to run preprocessing. Simply set *preprocessing=False* and *fitting=True* in the config file, and set the *gp_directory* to the folder where you have GPFiles and SliceInfoFile.txt files, separated into one folder per case. 

If you want to generate GPFiles yourself (i.e., not using biv-me preprocessing), but you don't know how to, the example GPFiles in `demo/guidepoints/example` can serve as reference for the required format.

Models will be generated as .txt files containing mesh vertex coordinates, .html plots for visualisation, and (optionally) .obj or .vtk files for LV endocardial, RV endocardial, and epicardial meshes.

### Running end-to-end pipeline (preprocessing and fitting)
If you specify in your config file to run both preprocessing and fitting, they will run in sequence as an end-to-end pipeline, such that biv-me models will be generated for each case for which there is DICOM data. When running as an end-to-end pipeline, there is no need to set the *gp_directory* for the fitting, as this will be automatically set as the *output_directory* of the preprocessing.

## Visualisation of models (and images)
Models can be visualised as .html plots which can be opened and interacted with inside your browser, or imported into a mesh visualisation software as .obj or .vtk objects to be viewed as a temporal sequence. The option to plot images alongside models in the .html plots is available to you in the config file (*include_images=True*), as is the option to export images as .vtk objects to display alongside fitted models (*export_images=True* - default is False) in a third-party software. Reviewing these settings to generate such visualisations can aid verification that the mesh reconstruction is faithful to the images.

After performing fitting for the example case, .html plots can be found in `demo/fitted-models/test-run/patient1/html`, and .vtk models and images can be found in the `demo/fitted-models/test-run/patient1/vtk-meshes` and `demo/fitted-models/test-run/patient1/vtk-images` folders respectively -- as long as the relevant config options have been set. 

Slice shifts (to correct for breath-hold misalignment) are automatically applied to the images to ensure that models and images are correctly registered. These slice shifts are derived from the fitting process, and are therefore unavailable if fitting is not performed. 

### Example .html plot (with images)
Contour types, individual mesh surfaces, and images can be interactively toggled, and rotation and zoom can be applied to inspect different areas of interest. 

![HtmlPlot1](images/html1.png)
![HtmlPlot2](images/html2.png)

### Example .vtk image and model sequence
By importing .vtk models and images into a mesh visualisation software such as Paraview, temporal sequences such as the following can be generated, analysed, and exported into video format. 

![VtkVis1](images/vis1.gif)
![VtkVis2](images/vis2.gif)


## Analysis of models
Several tools are provided for the analysis of biv-me models, including scripts for volume calculation, strain analysis (circumferential and longitudinal strains), and wall thickness measurement. 

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
| `-mdir MODEL_DIR`     | Specifies the path to the directory containing the biv-me models.                      |
| `-o OUTPUT_PATH`      | Specifies the directory where the output files will be saved.                                 |
| `-b BIV_MODEL_FOLDER` | Path to the folder containing the subdivision matrices for the models (default: `src/model`). |
| `-pat PATTERNS`       | The folder pattern to include for processing. You can use wildcards (default: `*`).           |
| `-p PRECISION`        | Sets the output precision (default: 2 decimal places).                                        |

#### **Example Usage** 
Example data is available in `demo/fitted-models/example`. To compute the volumes using this data, run the following command:

```python
cd src/bivme/analysis
python compute_volume.py -mdir ../../../demo/fitted-models/example -p 1 -o example_volumes
```

This will process the biv-me models in the `../../../demo/fitted-models/example` directory, compute the volumes with a precision of 1 decimal place, and save the results in the `example_volumes` directory. The volumes will be saved in the `lvrv_volumes.csv` file.

**Sample Output** <br> 
The output file will look like this:

| **Name**     | **Frame** | **LV Volume (lv_vol)** | **LV Mass (lvm)** | **RV Volume (rv_vol)** | **RV Mass (rvm)** | **LV Epicardial Volume (lv_epivol)** | **RV Epicardial Volume (rv_epivol)** |
|--------------|-----------|------------------------|-------------------|------------------------|-------------------|---------------------------------------|--------------------------------------|
| patient_1    | 0         | 172.6                 | 128.5           | 172.8                    | 53.8              | 295.0                                 | 224.0                               |
| patient_1    | 1         | 166.2                 | 129.3            | 172.4                | 54.1             | 289.3                                | 223.9                               |


### Calculating strains from models 

The script for calculating both global circumferential and global longitudinal strains of a mesh can be found in the `src/bivme/analysis` directory. Geometric strain is defined as the change in geometric arc length from ED to any other frame using a set of predefined points and calculated using the Cauchy strain formula. The global circuferential strains are calculated at three levels: apical, mid and basal. The global longitudinal strains are calculated on a 4ch and a 2ch view. They are calculated and stored in fractional form (i.e. -0.1 instead of -10%).

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
| `-mdir MODEL_DIR`     | Specifies the path to the directory containing the biv-me models.                      |
| `-o OUTPUT_PATH`      | Specifies the directory where the output files will be saved.                                 |
| `-b BIV_MODEL_FOLDER` | Path to the folder containing the subdivision matrices for the models (default: `src/model`). |
| `-pat PATTERNS`       | The folder pattern to include for processing. You can use wildcards (default: `*`).           |
| `-p PRECISION`        | Sets the output precision (default: 2 decimal places).                                        |
| `-ed ED_FRAME` | defines which frame is the ED frame. (default: 1st frame)


#### **Example Usage**
Example data is available in `demo/fitted-models/example`. To compute the circumferential strains using this data, run the following command:

```python
cd src/bivme/analysis
python compute_global_circumferential_strain.py -mdir ../../../demo/fitted-models/example -p 1 -o example_strains -ed 0
```

This will process the biv-me models in the `../../../demo/fitted-models/example` directory, compute the global circumferential strain with a precision of 1 decimal place, and save the results in the `example_strains` directory. The GCS will be saved in the `global_circumferential_strain.csv` file. The first frame will be used as ED. 

**Sample Output** <br>
The output file will look like this:

| **name**       | **frame** | **lv_gcs_apex** | **lv_gcs_mid** | **lv_gcs_base** | **rvfw_gcs_apex** | **rvfw_gcs_mid** | **rvfw_gcs_base** | **rvs_gcs_apex** | **rvs_gcs_mid** | **rvs_gcs_base** |
|------------|-------|-------------|------------|-------------|----------------|---------------|----------------|---------------|--------------|---------------|
| patient_1 | 0     | 0           | 0          | 0           | 0              | 0             | 0              | 0             | 0            | 0             |
| patient_1 | 1     | -0.006071119	| -0.00602047	| -0.022775424	| -0.002317497	| -0.005453306	| 0.015503876 |	0.002590674	| -0.00312989	|0.010297483



### Calculating wall thickness from models
The script computing the wall thickness can be found in src/bivme/analysis. Wall thickness is calculated on binary 3D images using [pyezzi](https://pypi.org/project/pyezzi/) for both LV and RV separately. The septal wall is included in the LV calculation and excluded from the RV. 

To run the `compute_wall_thickness.py` script, use the following command:

```bash
usage: compute_wall_thickness.py [-h] [-mdir MODEL_DIR] [-o OUTPUT_PATH] [-b BIV_MODEL_FOLDER] [-pat PATTERNS] [-r VOXEL_RESOLUTION] [-s SAVE_SEGMENTATION_FLAG]
 ```

| **Argument**          | **Description**                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------- 
| `-h, --help`          | Displays the help message and exits.                                                          |
| `-mdir MODEL_DIR`     | Specifies the path to the directory containing the biv-me models.                      |
| `-o OUTPUT_PATH`      | Specifies the directory where the output files will be saved.                                 |
| `-b BIV_MODEL_FOLDER` | Path to the folder containing the subdivision matrices for the models (default: `src/model`). |
| `-pat PATTERNS`       | The folder pattern to include for processing. You can use wildcards (default: `*`).           |
| `-r VOXEL_RESOLUTION`        | Voxel resolution to compute the masks.                                        |
| `-s SAVE_SEGMENTATION_FLAG` | Boolean flag indicating whether to save 3D masks


#### **Example Usage**
Example data is available in `demo/fitted-models/example`. To compute the wall thickness using this data, run the following command:

```
python compute_wall_thickness.py -mdir ../../../demo/fitted-models/example -o example_thickness
```

This will process the biv-me models in the `../../../demo/fitted-models/example` directory, compute the wall thickness at a resolution of 1mm, and save the results in the `example_thickness` directory. Wall thickness is sampled and saved at the location of each vertex and can be visualised in Paraview as vertex color or in 3D slicer.

Adding the `-s` flag to the above command will also generate 4 extra nifti files per model: 2 3D masks with background=0, cavity=1, and wall=2 (`labeled_image_lv*.nii` and `labeled_image_lv*.nii`) and 2 3D mask containing thickness values at each voxel (`lv_thickness*.nii` and `rv_thickness*.nii`).

```
python compute_wall_thickness.py -mdir ../../../demo/fitted-models/example -o example_thickness -s
```

#### **Sample Output**
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
| `-config CONFIG_FILE`     | Path to config file describing which modules to run and their associated parameters.                   

The config file should be the one used to fit the original models. Refitted models will be saved in config["output_fitting"]["output_directory"].

## FAQs
### *How do I update my deep learning models?*

If you have already installed biv-me and new deep learning models have been released since you installed it, you can simply pull the latest version of biv-me from the main branch...

```bash
git pull
```

...and rerun the command to install the git submodule.

```bash
git submodule update --init
```

This will update both biv-me and the deep learning models. If you would rather only update the deep learning models to the latest version without updating biv-me, you can run

```bash
git submodule update --init --remote
```

### *How do I access a specific version of the deep learning models?*

If you want to access any previous version of the deep learning models, you can visit the [deep learning model repository](https://github.com/UOA-Heart-Mechanics-Research/biv-me-dl-models) to find the tag for that version (e.g. v1.0).

To roll back the models, you need to checkout the submodule at the version you want in your local clone of biv-me. To do so, type the following into your terminal, where 'tag' is the tag for the version of the models you want to roll back to.

```bash
cd src/bivme/preprocessing/dicom/models
git checkout 'tag'
```

For example, if I wanted to roll back to the v1.0 models, I would enter

```bash
git checkout v1.0
```

### *The code doesn't read in some or all of my images. There is nothing wrong with my images, so why might this be happening?*
There are multiple possible explanations. 

One possible reason is that your DICOMs are stored in a remote server accessed by an unstable or intermittent network connection, causing occasional dropouts and failures to read certain images. If possible, biv-me should be run locally or through a wired connection to ensure this does not happen. 

Another reason why some or all of your images cannot be read is that your DICOMs have a certain type of image compression that is not supported by pydicom. We have encountered this problem in the past and have since made adjustments, namely by incorporating pylibjpeg into the biv-me environment. However, we have not seen every possible form of image compression. If you think this problem might be occurring with your data, reach out to us with an example and we will be happy to look into it.

Another explanation is that your images are being filtered out. We use the series description tag of the DICOMs to infer which series are cines and which are not. This works 99% of the time, but it may not suit your dataset. You can review the string keys used to exclude non-cine series in `src/bivme/preprocessing/dicom/extract_cines.py` and change them as needed.

### *Why does the code run slowly on my Mac?*
The preprocessing pipeline uses 3D nnU-Net models to perform automatic segmentation. Unfortunately, [Apple silicon chips (M1-) with MPS do not currently support 3D convolutions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md), so segmentation is carried on CPU instead, and may take a long time to process. If possible, consider running preprocessing on a CUDA enabled device.   


## Contribution - notation
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
This work is based on contributions by **Laura Del Toso**, **Esther Puyol-Anton**, **Sachin Govil**, **Brendan Crabb**, **Anna Mîra**, **Liandong Lee**, **Richard Burns**, **Devran Ugurlu**, **Ayah Elsayed**, **Grant Roberts**, **Debbie Zhao**, **Joshua Dillon**, and **Charlène Mauger**. 

We gratefully acknowledge the study participants who contributed their data for development and validation of the biv-me pipeline.

## Contact
For questions or issues, please open an issue on GitHub or contact [joshua.dillon@auckland.ac.nz](joshua.dillon@auckland.ac.nz) or [charlene.1.mauger@kcl.ac.uk](charlene.1.mauger@kcl.ac.uk) 
