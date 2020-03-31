# common
import sys
import os.path as op

# basic
import numpy as np
import pandas as pd
import xarray as xr
from time import time

t0 = time()

# dev library 
sys.path.insert(0, op.join(op.dirname(__file__), '..'))

# calval wrap module
from lib.calval import CalVal

# --------------------------------------------------------------------------- #
# data
print('--------------------------------------------------------')
print('Reading data...')
print('--------------------------------------------------------')

p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))

buoy       =  pd.read_pickle(op.join(p_data, 'Bilbao-Vizcaya Ext.pkl'))
csiro      =  pd.read_pickle(op.join(p_data, 'csiro_dataframe.pkl'))
satellite  =  xr.open_dataset(op.join(p_data, 'satellite_dataset.nc'))
bat_spain  =  np.loadtxt(op.join(p_data, 'SPAIN_2020_bath.dat'))
bat_gebco  =  xr.open_dataset(op.join(p_data, 'GEBCO_2019_bath.nc'))

print('--------------------------------------------------------')
print('Initializing the constructor...')
print('--------------------------------------------------------')

calval_case = CalVal(buoy, csiro, satellite, 'CSIRO', bat_spain, bat_gebco)

print('Time wasted initializing the constructor: ' + str((time()-t0)/60) + ' m')

calval_case.region_map()

calval_case.buoy_comparison('raw')
calval_case.buoy_comparison('sat_corr')
calval_case.buoy_comparison('buoy_corr')

calval_case.buoy_validation('raw')
calval_case.buoy_validation('sat_corr')
calval_case.buoy_validation('buoy_corr')

print('Time of the script: ' + str((time()-t0)/60) + ' m')

calval_case.hindcast_sat_corr.to_pickle(op.join(p_data, 'csiro_dataframe_sat_corr.pkl'))
calval_case.hindcast_buoy_corr.to_pickle(op.join(p_data, 'csiro_dataframe_buoy_corr.pkl'))