# common
import sys
import os
import os.path as op

# basic
import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import loadmat
from time import time

t0 = time()

# dev library 
sys.path.insert(0, op.join(op.dirname(__file__), '..'))

# calval wrap module
from lib.calval import CalVal

# --------------------------------------------------------------------------- #
# data
p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))

buoy       =  pd.read_pickle(p_data, 'Bilbao-Vizcaya Ext.pkl')
csiro      =  pd.read_pickle(p_data, 'csiro_dataframe.pkl')
satellite  =  xr.open_dataset(p_data, 'satellite_dataset.nc')
bat_spain  =  np.loadtxt(p_data, 'SPAIN_2020_bath.dat')
bat_gebco  =  xr.open_dataset(p_data, 'GEBCO_2019_bath.nc')

calval_case = CalVal(buoy, csiro, satellite, 'CSIRO', bat_spain, bat_gebco)
calval_case.region_map()

#print('Time wasted initializing the constructor: ' + str(time()-t0))
#
#calval_case.buoy_comparison('raw')
#calval_case.buoy_comparison('sat_corr')
#calval_case.buoy_comparison('buoy_corr')
#
#calval_case.buoy_validation('raw')
#calval_case.buoy_validation('sat_corr')
#calval_case.buoy_validation('buoy_corr')
#
#print('Time of the script: ' + str(time-t0()))