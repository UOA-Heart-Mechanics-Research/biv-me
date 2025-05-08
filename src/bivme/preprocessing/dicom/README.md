<div align="center">

# Automatic DICOM preprocessing pipeline

</div>

This code reads in CMR DICOM files and generates GPFiles for personalised biventricular mesh fitting.

## Usage

To run the preprocessing code, you can use the following command.

```python
cd src/bivme
python main.py -config ../configs/config.toml
```

Make sure to set up the config file in advance, and make sure that preprocessing is set to true. If you don't want to run fitting right away, you can skip filling the fields related to fitting. Otherwise, running the main.py script with both preprocessing and fitting enabled will generate biventricular models for each case in your source directory into the output directory you specify. 

You can also run the preprocessing from a Jupyter notebook, located at `src\bivme\preprocessing\dicom\run_preprocessing_pipeline_interactive.ipynb`. This notebook runs the preprocessing case by case. It is particularly useful if you would like some tighter supervision over certain aspects, such as the view prediction. It is well documented, and a good place to start if you are feeling uncertain as to how everything works. 

If you want to run the Jupyter notebook, you will need to activate ipython. Use this command below to activate the ipython kernel for the bivme311 environment.

```
conda install -n bivme311-clean ipykernel --update-deps --force-reinstall
```

For more information, refer to the main README. 

## Troubleshooting model performance
Though we are confident in the robustness of our deep learning models, they may not work perfectly for your data. If you find that the segmentation or view selection models perform poorly for your data, reach out to us at [joshua.dillon@auckland.ac.nz](joshua.dillon@auckland.ac.nz) or [charlene.1.mauger@kcl.ac.uk](charlene.1.mauger@kcl.ac.uk) and let us know what kind of data you are using. We are actively developing these models and always looking for ways to enhance their generalisability across vendors, protocols, centres, and patient demographics.


## Credits

Special thanks are given to Sachin Govil, Brendan Lee, Yu Deng, and Ayah Elsayed for their contributions to development of the preprocessing pipeline. 
