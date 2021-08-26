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

# Csiro data
csiro      =  pd.read_pickle(op.join(p_data, 'hindcast', 
                                     'csiro_dataframe_can.pkl'))
print(csiro.info())

# Satellite data, see extract_csiro.py
# An example for satellite boundary box is:
# 43.8, 44.2, 356.2, 356.6
satellite  =  xr.open_dataset(op.join(p_data, 'satellite', 
                                      'satellite_dataset_can.nc'))
print(satellite)

# Buoy data
buoy       =  xr.open_dataset(op.join(p_data, 'buoy', 
                                      'bilbao_offshore_buoy.pkl'))
print(buoy)

print('--------------------------------------------------------')
print('Initializing the constructor...')
print('--------------------------------------------------------')

calval_case = calval.CalVal(hindcast=csiro, 
                            hindcast_longitude=csiro_lon,
                            hindcast_latitude=csiro_lat,
                            satellite=satellite,
                            buoy=(True,buoy.to_dataframe()),
                            buoy_longitude=buoy.longitude,
                            buoy_latitude=buoy.latitude,
                            buoy_corrections=False)

# if buoy data does not exist, just delte the buoy and buoy_correction
# attributes, and then comment the methods that use the buoy information

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
                                         

