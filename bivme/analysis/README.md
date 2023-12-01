# Analysis

Author: Laura Dal Toso                    

Date: 21/10/2022

------------------------------------------
Scripts for the analysis of the models generated using the BiV_Modelling framework. These scripts can be used to measure the strain, mass and volume of LV and RV. 

## Contents

- compute_circum_global_strain.py: computes the circumferential global strain from models
- compute_global_longitudinal_strain.py: computes the global longitudinal strain from models
- Mass_volume: measures the LV and RV mass and volume from biventricular models
- ModifyExistingCsv: modifies an existing spreadsheet containing strain/mass/volume, by replacing old values with new ones.
- DeleteOutliers: deletes outliers from a spreadsheet based on interquartile range.

## Usage

**Strain**

Once the control points of the biventricular models have been generated and saved (usually in .txt files), you can perform strain analysis using compute_circum_global_strain.py and compute_global_longitudinal_strain.py. These two scripts are sctrutured in the same way, so the usage is similar. 

At the moment, the scripts are structured to print all data relative to a single patient in a single line. This works well when the analysis is performed on a small numer of frames, but it may need to be changed when applying the script to a time series composed of many frames.

**Mass and Volume**

You can use Mass_volume.py for the measurement of LV and RV mass and volume from biventricular models. The input is the model file, the output is a csv file containing all the desidered mass and volume measurements at ES and ED. This script is to be applied only on Biobank data, where ES and ED frame numbers always fall in the same range. If you want to apply it on a different dataset, change the definition of ed and es frame. 

At the moment, the script is structured to print all data relative to a single patient in a single line. This works well when the analysis is performed at ES and ED, but it may need to be changed when applying the script to a time series composed of many frames.

**Modify an existing spreadhseet**

ModifyExistingCsv can be used if an existing spreadsheet containing strain, mass or volume has to be changes. This can happen if some of teh biventricular models were wrong, or if there was an error in the computation of a strain/mass/volume entry. 


**Delete outliers**

In preparation for further analysis, outliers can be deleted from the csv files by running DeleteOutliers.py. This script used an interquartile range criterion to delete outliers. 

## Credits

Anna Mira
