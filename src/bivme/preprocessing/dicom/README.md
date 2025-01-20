<div align="center">

# Automatic DICOM preprocessing pipeline

</div>

This code reads in DICOM files and generates GPFiles for personalised biventricular mesh fitting for the entire cardiac cycle.

## Usage

-----------------------------------------------

## Import libraries
This preprocessing pipeline utilises PyTorch and nnU-Net. The default biv-me conda environment currently doesn't install either of these for you. To set these up, activate the biv-me conda environment, like so:

```
conda activate bivme311
```

Then, find the PyTorch right version for your GPU and OS ([here](https://pytorch.org/get-started/locally/)) and install it as described on the website.

After PyTorch has been installed, install nnU-Net like so:

```
pip install nnunetv2
```

### Before running
The preprocessing pipeline expects DICOM files to be organised in a certain way. There are three key things to get right with the DICOMs before running the code. 

Firstly, there should be no non-cine images or non-cardiac views. My job is hard enough as it is. Some day I'll add a pre-preprocessing step, but not today.

Secondly, images should be separated into folders by case, like so:

    ```
    input_path   
    └─── case1
        │─── *.dcm
    └─── case2
        │─── *.dcm
    └─── ...
    ```

Thirdly, all images should be converted to .dcm format.

Once your DICOMs are organised, set up the directories in the config file located at src/bivme/preprocessing/dicom/configs/preprocess_config.toml. If you wish, you can create a new config file for each time you want to make changes. Just make sure to update the argument you call the code with.

### Preprocess DICOM files for fitting
The main script for running the preprocessing pipeline mesh fitting can be found in src/bivme/preprocessing/dicom. This runs the pipeline on all images stored within the main DICOM folder you provide it. The output is a set of GPFiles for each frame of each case, which can be used for fitting.

```
usage: run_pipeline.py [-h] [-config CONFIG_FILE]

Preprocess DICOM files for fitting

options:
  -h, --help            show this help message and exit
  -config CONFIG_FILE, --config_file CONFIG_FILE
                        Config file containing preprocessing parameters

```

You can also run the preprocessing from a Jupyter notebook, in the same directory, named run_pipeline_interactive.ipynb. This notebook runs case by case. It is particularly useful if you would like some tighter supervision over certain aspects, such as the view prediction. 

## Outputs
The outputs of the preprocessing pipeline are a SliceInfoFile.txt, containing slice metadata, and GPFile_*.txt - one per frame. To generate biventricular models from these GPFiles, refer to the main README.


Credits
------------------------------------
Based on work by: Sachin Govil, Brendan Lee, and Joshua Dillon
