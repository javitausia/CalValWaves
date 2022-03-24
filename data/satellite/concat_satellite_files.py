import os
import xarray as xr
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
for root, dirs, files in os.walk(
    os.path.join(
        os.getcwd(), 'sat_netcdf_files_raglan'
    ), topdown=True
):
    os.chdir(root)
    for name in files:
        try:
            sat_file = xr.open_dataset(name)
        except:
            print('Be careful with not .nc files!!')
            continue
        sat_files.append(sat_file)
        cs += 1
        if(cs%step==0):
            sat_datasets.append(xr.merge(sat_files,compat='override'))
            sat_files = []
            print('First ' + str(cs) + ' cases read and joined in ' + str(time()-t0) + ' s')
            t0 = time()
sat_datasets.append(xr.merge(sat_files,compat='override'))
sat_dataset = xr.merge(sat_datasets,compat='override') # TODO: check override
print('--------------------------------------------------------')
print('All files joined...')
print('--------------------------------------------------------')
sat_dataset.to_netcdf(os.path.join('..', 'satellite_dataset_raglan.nc'))