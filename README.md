# Soft Sensor OMP

## Data

*USGS_data* contains ADVM, SSC, Q of 3 sites of a USGS study conducted in Minnesota

*GVR_data_code* folder for output kernel covariance matrix to be in .mat form


## Scripts

*SCB_conversion.py* is the method used by the USGS study to calculated SAC from ADVM

*file_opening.py* is made to open the csv files easily and have some datetime properties

*GP_func.py* are generalized functions for running GP regressions in the main jupyter notebook

*soft_sensing.ipynb* notebook for all analysis done for paper. Reads in USGS data, runs a linear regression and GP regression for each site. Creates the kernel covariance matrix output to put into soft sensor sfo matlab package. Runs OMP and random sample selection strategies and computes RMSE for all models. Finally predicts SSC time series off of selected models
