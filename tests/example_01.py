# common
import sys
import os.path as op

# basic
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

# Path to the data (this is the data I used)
p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))

# Buoy data
buoy       =  pd.read_pickle(op.join(p_data, 'buoy', 
                                     'Bilbao-Vizcaya Ext.pkl'))
print(buoy.info())

# Csiro data
csiro      =  pd.read_pickle(op.join(p_data, 'hindcast', 
                                     'csiro_dataframe.pkl'))
print(csiro.info())

# Satellite data, see extract_csiro.py
# An example for satellite boundary box is:
# 43.8, 44.2, 356.2, 356.6
satellite  =  xr.open_dataset(op.join(p_data, 'satellite', 
                                      'satellite_dataset.nc'))
print(satellite)

print('--------------------------------------------------------')
print('Initializing the constructor...')
print('--------------------------------------------------------')

calval_case = CalVal(buoy, csiro, satellite)

print('Time wasted initializing the constructor: ' 
      + str((time()-t0)/60) + ' m')


calval_case.buoy_comparison('raw')
calval_case.buoy_comparison('sat_corr')
calval_case.buoy_comparison('buoy_corr')

calval_case.buoy_validation('raw')
calval_case.buoy_validation('sat_corr')
calval_case.buoy_validation('buoy_corr')


print('Time of the script: ' + str((time()-t0)/60) + ' m')

calval_case.hindcast_sat_corr.to_pickle(op.join(p_data, 'hindcast',
                                        'csiro_dataframe_sat_corr.pkl'))
calval_case.hindcast_buoy_corr.to_pickle(op.join(p_data, 'hindcast',
                                         'csiro_dataframe_buoy_corr.pkl'))