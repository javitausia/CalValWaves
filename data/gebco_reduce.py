#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:48:33 2020

@author: tausiaj
"""

import xarray as xr

gebco = xr.open_dataset('GEBCO_2019.nc')
gebco = gebco.sel(lat=slice(42,45)).sel(lon=slice(-8,-1))
gebco.to_netcdf('GEBCO_2019_bath.nc')