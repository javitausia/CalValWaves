import xarray as xr
import netCDF4
import pandas as pd
import numpy as np
import os
from time import time

# Extract the .txt file from the folder

t0 = time()
sat_files = []
sat_datasets = []
cs = 0
step = 20
print('--------------------------------------------------------')
print('Concatinating satellite files in steps of ' + str(step))
print('--------------------------------------------------------')
for root, dirs, files in os.walk(os.path.join(os.getcwd(), 'sat_netCDFs'), 
                                 topdown=True):
    os.chdir(root)
    for name in files:
        sat_file = xr.open_dataset(name)
        sat_files.append(sat_file)
        cs += 1
        if(cs%step==0):
            sat_datasets.append(xr.merge(sat_files))
            sat_files = []
            print('First ' + str(cs) + ' cases read and joined in ' + str(time()-t0) + ' s')
            t0 = time()
sat_datasets.append(xr.merge(sat_files))
sat_dataset = xr.merge(sat_datasets)
print('--------------------------------------------------------')
print('All files joined...')
print('--------------------------------------------------------')
sat_dataset.to_netcdf(os.path.join('..', 'satellite_dataset.nc'))
