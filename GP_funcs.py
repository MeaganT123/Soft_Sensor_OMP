#importing libraries
import numpy as np
import math as math
import random

#For GP
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct

'''
===================================================================================
@desc - Runs gaussian process on SSC from SAC
@param - 
    {Array} SAC_m Sediment Attenuation Coefficient training data matching time series of SSC_m
    {Array} SSC_m Suspended Sediment Concentration training data grab matching time series of SAC_m
    {number} p Number of points wanted for GP estimate
    {Array} [SAC_range = [None,None]] option to put range of SAC for bin creation '[min, max]',
        will automatically calculate from max and min
    {Boolean} [SAC_log = False] if want to make output SAC log of inputed SAC data
    {Boolean} [SSC_log = False] if want to make output SSC log of inputed SSC data
@returns - 
    {Array} SAC_pred - SAC range for model prediction
    {Array} SSC_pred - Predicted SSC values based on model
    {Array} std - standard deviation of GP at each point
    {Array} GP_score - GP score based on fit
    {Array} GP_logmarg - GP log marginal value of model
    gaussian_proccess - trained model
@potential future upgrades - 
        allow array of X values (multivariable GP fit)
        Round min and max to a set sig fig (with log values general floor() or ceil() greatly out of typical range
===================================================================================
'''
def GP(SAC_m, SSC_m, p, SAC_range = [None,None], SAC_log=False, SSC_log=False):
    #assuming don't want log value (and pass regular values)
    kernel = RBF() + WhiteKernel()

    # Calculate range for model run 
        #safety measure if not put in correct order and if passed as list
    if SAC_range != [None,None]:
        min_SAC = np.array(SAC_range).min()
        max_SAC = np.array(SAC_range).max()
    else:
        min_SAC = SAC_m.min()
        max_SAC = SAC_m.max()    

    #Set variables for training
    X = np.array(SAC_m).reshape(-1,1)
    y = SSC_m #hand made for scale 2
    
    #Change to log based if output desired is in log and calucate range
    if SAC_log == True:
        min_SAC = np.log10(min_SAC)
        max_SAC = np.log10(max_SAC)
        X = np.log10(np.array(SAC_m).reshape(-1,1))
    if SSC_log == True:
         y = np.log10(SSC_m) 

    #Make range for SAC prediction
    SAC_pred = np.linspace(min_SAC,max_SAC,p)

    #Create model from training data
    # https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
    gaussian_proccess = GaussianProcessRegressor(kernel=kernel, normalize_y = True)
    gpr = gaussian_proccess.fit(X,y)
    gaussian_proccess.kernel_
    GP_score = gpr.score(X,y)
    GP_logmarg = gpr.log_marginal_likelihood()
    
    #Make predicted model from kernel
    [Y_pred, std] = gpr.predict(SAC_pred.reshape(-1, 1) , return_std = True)

    return (SAC_pred, Y_pred, std, GP_score, GP_logmarg, gaussian_proccess)


'''
===================================================================================
@desc - Selects samples to evenly fill bins then runs gaussian process on subset of data
@param - 
    {Array} SAC_m Sediment Attenuation Coefficient training data matching time series of SSC_m
    {Array} SSC_m Suspended Sediment Concentration training data grab matching time series of SAC_m
    {Array} Q_m Flow rate measurements matching time series of SSC_m
    {Array} date_time_m time stamp for measurements
    {number} r number of random samples runs
    {number} n number of samples in subset
    {number} p Number of points wanted for GP prediction
    {number} [seed_num = 0] set seed number for random generation for consitent runs
    {Boolean} [seed_lock = True] default to use same seed between calls, False would ignore 'seed_num'
    {Boolean} [SAC_log = False] if want to make output SAC log of inputed SAC data
    {Boolean} [SSC_log = False] if want to make output SSC log of inputed SSC data
    {Array} [SAC_range = [None,None]] option to put range of SAC for bin creation '[min, max]',
@returns - 
    {Array} SAC_r - Random subset of SAC measurements
    {Array} SSC_r - Random subset of SSC measurements
    {Array} date_time_r - corresponding timestamp for random subset
    {Array} Q_r - corresponding flow rates for random subset
    {Array} SAC_pred - SAC range for model prediction
    {Array} Y_pred - Predicted SSC values based on model
    {Array} std - standard deviation of GP at each point
    {Array} GP_score - GP score based on fit
    {Array} GP_logmarg - GP log marginal value of model
@potential future upgrades - 
        Round min and max to a set sig fig (with log values general floor() or ceil() greatly out of typical range
        Change to work with data frames so don't have to deal with matching indicies
        Could change so array of n and r to have changing number of samples and runs per number of samples
        allow multiple parameters (not just SAC) to be passed
===================================================================================
'''
def Rand_m(SAC_m, SSC_m, Q_m, date_time_m, r, n, p, seed_num = 0, seed_lock = True, SAC_log = False, SSC_log = False, SAC_range = [None,None]):
    
    if seed_lock == True:
        random.seed(seed_num)
    
    #initial arrays to store all runs
    SAC_r, SSC_r, Q_r, date_time_r, \
    SAC_pred, Y_pred, std, GP_score, GP_logmarg, GP_model = ([None] * r for i in range(9))
    
    i=0 #create subset of data for all subsets i
    while i < r:
        L = np.linspace(0,len(SAC_m)-1,len(SAC_m)).astype(int)
        ind = random.sample(list(L), n)
        
        #Store run i random subset
        SAC_r[i] = SAC_m[ind]
        SSC_r[i] = SSC_m[ind]
        Q_r[i] = Q_m[ind]
        date_time_r[i] = date_time_m[ind]

        SAC_pred[i], Y_pred[i], std[i], GP_score[i], GP_logmarg[i], GP_model[i] = GP(SAC_r[i], SSC_r[i], p, SAC_log = SAC_log, SSC_log = SSC_log, SAC_range = SAC_range)
        i += 1
        
    #change output SAC_r, SSC_r if log is desired output
    if SAC_log == True:
        SAC_r = np.log10(SAC_r)
    if SSC_log == True:
        SSC_r = np.log10(SSC_r) 
            
    return(SAC_r, SSC_r, date_time_r, Q_r, SAC_pred, Y_pred, std, GP_score, GP_logmarg, GP_model)



# ===================================================================================
# *
# **
# ***
# ****
# *****
# ---------------------------- Statistics Calculations ----------------------------
# *****
# ****
# ***
# **
# *
# ===================================================================================


'''
===================================================================================
@desc - Calculates Mean Squared Error of multiple vectors at once
@param - 
    {Array} y_true 1D array of values taking as the "truth" to compare to
    {Array} preds 2D array of arrays of predicted values
@returns - 
    {Array} Returns 1D array of mean squared error, each index goes to corresponding prediction array
===================================================================================
'''
def mmse (Y_true, preds):
    mse_a = [None] * len(preds)
    Y = np.array(Y_true)
    x = np.array(preds)
    i = 0
    while (i < len(preds)):
        mse_a[i] = np.square(np.subtract(Y,x[i])).mean()
        i += 1

    return mse_a

'''
===================================================================================
@desc - Calculates Mean Average Error of multiple vectors at once
@param - 
    {Array} y_true 1D array of values taking as the "truth" to compare to
    {Array} preds 2D array of arrays of predicted values
@returns - 
    {Array} Returns 1D array of mean absolute error, each index goes to corresponding prediction array
===================================================================================
'''
def mmae (Y_true, preds):
    #MAE for bins
    mae_a = [None] * len(preds)
    Y = np.array(Y_true)
    x = np.array(preds)
    i = 0
    while (i < len(preds)):
        mae_a[i] = np.mean(np.abs(Y - x[i]))
        i += 1

    return mae_a

