###############################################################################
######            JAVIER TAUSÍA HOYAL                                    ######
######            20 de Marzo de 2020                                    ######
###############################################################################

# common
import sys
import os
import os.path as op

# basic
import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import loadmat

# plotting
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import cmocean
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from mpl_toolkits.basemap import Basemap

# additional
from time import time
from datetime import datetime as dt
from datetime import timedelta as td

# additional*
import scipy.stats as stats
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error
from sklearn import datasets, linear_model, metrics
import statsmodels.api as sm

#-----------------------------------------------------------------------------#
# Metric and accuracy functions
def rmse(pred, tar):
    return np.sqrt(((pred - tar) ** 2).mean())

def bias(pred, tar):
    return sum(pred - tar) / len(pred)

def si(pred, tar):
    S = pred.mean()
    O = tar.mean()
    return np.sqrt(sum(((pred-S) - (tar-O)) ** 2) / ((sum(tar ** 2))))

#-----------------------------------------------------------------------------#
# Direc selection function
def create_vec_direc(waves, direcs):
    data = np.zeros((len(waves), 16))
    for i in range(len(waves)):
        if (direcs[i] < 0):
            direcs[i] = direcs[i] + 360
        if (direcs[i] > 0 and waves[i] > 0):
            if (direcs[i] >= 0 and direcs[i] < 22.5):
                data[i, 0] = waves[i]
            elif (direcs[i] >=22.5 and direcs[i] < 45):
                data[i, 1] = waves[i]
            elif (direcs[i] >=45 and direcs[i] < 67.5):
                data[i, 2] = waves[i]
            elif (direcs[i] >=67.5 and direcs[i] < 90):
                data[i, 3] = waves[i]
            elif (direcs[i] >=90 and direcs[i] < 112.5):
                data[i, 4] = waves[i]
            elif (direcs[i] >=112.5 and direcs[i] < 135):
                data[i, 5] = waves[i]
            elif (direcs[i] >=135 and direcs[i] < 157.5):
                data[i, 6] = waves[i]
            elif (direcs[i] >=157.5 and direcs[i] < 180):
                data[i, 7] = waves[i]
            elif (direcs[i] >=180 and direcs[i] < 202.5):
                data[i, 8] = waves[i]
            elif (direcs[i] >=202.5 and direcs[i] < 225):
                data[i, 9] = waves[i]
            elif (direcs[i] >=225 and direcs[i] < 247.5):
                data[i, 10] = waves[i]
            elif (direcs[i] >=247.5 and direcs[i] < 270):
                data[i, 11] = waves[i]
            elif (direcs[i] >=270 and direcs[i] < 292.5):
                data[i, 12] = waves[i]
            elif (direcs[i] >=292.5 and direcs[i] < 315):
                data[i, 13] = waves[i]
            elif (direcs[i] >=315 and direcs[i] < 335.5):
                data[i, 14] = waves[i]
            elif (direcs[i] >=335.5 and direcs[i] < 360):
                data[i, 15] = waves[i]           
    return data
    
#-----------------------------------------------------------------------------#
# Time calibration
def calibration_time(sat, hind, sh = True, min_time = 2):
    # create the empty arrays
    times_sat = np.array([], dtype = 'datetime64')
    times_hind = np.array([], dtype = 'datetime64')
    # perform the calibration
    if sh:
        for i in range(len(sat)):
            diff = np.min(abs(hind - sat[i]))
            if diff < np.timedelta64(min_time, 'h'):
                min_index = np.argmin(abs(hind - sat[i]))
                times_hind = np.append(times_hind, hind[min_index])
                times_sat = np.append(times_sat, sat[i])
    else:
        for i in range(len(hind)):
            diff = np.min(abs(sat - hind[i]))
            if diff < np.timedelta64(min_time, 'h'):
                min_index = np.argmin(abs(sat - hind[i]))
                times_sat = np.append(times_sat, sat[min_index])
                times_hind = np.append(times_hind, hind[i])
    # return
    return times_sat, times_hind

