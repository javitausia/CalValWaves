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

# custom
from .functions import rmse, bias, si, create_vec_direc, calibration_time


# Calibration-validation class
class CalVal(object):
    
    
    def __init__(self, buoy, hindcast, sat, hind, bath_spain, bath_gebco):
        """ Initializes the class with all the necessary attributes that
            will be used in the different methods
            ------------
            Parameters
            buoy: Buoy data as a dataframe
            hindcast: Hindcast data as a dataframe
            sat: Satellite data as a netCDF (see extract_satellite.py)
            hind: Name of hindcast used (CSIRO or ERA5)
            bath: Bathymetry that will be used to plot the map
            ------------
            Returns
            Attributes initialized and plots for both calibrations, the one
            done with the buoy and the other one performed with satellite
            data. The parameters for the calibrations are also stored
        """
        
        print('Initializing data to calibrate... \n ')
        
        self.buoy                 =    buoy
        self.hindcast             =    hindcast
        self.possible_to_correct  =    np.where(hindcast['Hs_'+hind+'_cal'] > 0.01)[0]        
        self.hind_to_corr         =    hindcast.iloc[self.possible_to_correct]
        self.sat                  =    sat
        self.hind                 =    hind
        self.hindcast_sat_corr, self.params_sat_corr    =    self.calibration('sat')
        self.hindcast_buoy_corr, self.params_buoy_corr  =    self.calibration('buoy')
        self.bath_spain           =    bath_spain
        self.bath_gebco           =    bath_gebco
    
    
    def calibration(self, calibration_type):
        """ Calibrates hindcast with satellite or buoy data. This calibration
            is performed using a linear regression and selecting only those
            parameters that are representative.
            ------------
            Parameters
            calibration_type: Type of calibration to be done (sat, buoy)
            ------------
            Returns
            Corrected data and calculated params
        """
        
        # Initializes satellite data to calibrate
        if calibration_type=='sat':
            print('Satellite box values: ')
            ini_lat = float(input('South latitude: '))
            end_lat = float(input('North latitude: '))
            ini_lon = float(input('West latitude: '))
            end_lon = float(input('East latitude: '))
            print(' \n ')
            hs_calibrate, calibration = self.satellite_values(ini_lat=ini_lat, 
                                                              end_lat=end_lat,
                                                              ini_lon=ini_lon, 
                                                              end_lon=end_lon)
            title = 'Satellite'
        elif calibration_type=='buoy':
            calibration_tot = pd.concat([self.buoy, self.hind_to_corr], 
                                        axis=1)
            calibration = calibration_tot.iloc[calibration_tot['Hs_Buoy']\
                                               .notna().values]
            hs_calibrate = calibration['Hs_Buoy'].values
            title = 'Buoy'
        else:
            message = 'Not a valid value for calibration_type'
            return message
        
        print(calibration)
        print(' \n ')
        
        # Construct matrices to calibrate
        print('--------------------------------------------------------')
        print(title.upper() + ' CALIBRATION')
        print('-------------------------------------------------------- \n ')
        print('Constructing matrices and calibrating... \n ')
        
        Hsea    = create_vec_direc(calibration['Hsea_'+self.hind], \
                                   calibration['Dirsea_'+self.hind])
        Hswell1 = create_vec_direc(calibration['Hswell1_'+self.hind], \
                                   calibration['Dirswell1_'+self.hind])
        Hswell2 = create_vec_direc(calibration['Hswell2_'+self.hind], \
                                   calibration['Dirswell2_'+self.hind])
        Hswell3 = create_vec_direc(calibration['Hswell3_'+self.hind], \
                                   calibration['Dirswell3_'+self.hind])
        Hs_ncorr_mat = np.concatenate([Hsea**2, Hswell1**2 + \
                                       Hswell2**2 + Hswell3**2], 
                                      axis=1)
        Hs_ncorr = np.sqrt(np.sum(Hs_ncorr_mat, axis=1))
        #---------------------------------------------------------------------#
        print('Value to set the umbral for not enough data to calibrate, ')
        print('this value can be set to 0.01, 0.02 or 0.03: ' )
        th_ne = float(input('----- Threshold ----- : '))
        print(' \n ')
        #---------------------------------------------------------------------#
        nedata = np.where(np.mean(Hs_ncorr_mat, axis=0) < th_ne)[0]
        reg = linear_model.LinearRegression() 
        hs_calibrate_2 = hs_calibrate**2
        reg.fit(Hs_ncorr_mat, hs_calibrate_2)
        X = sm.add_constant(Hs_ncorr_mat)
        est = sm.OLS(hs_calibrate_2, X)
        est2 = est.fit()
        params = np.array([], dtype = float)
        for p in range(1, len(est2.params)):
            if (est2.pvalues[p] < 0.05 and reg.coef_[p-1] > 0):
                params = np.append(params, reg.coef_[p-1])
            else:
                params = np.append(params, 1.0)
        params[nedata] = 1.0
        paramss = np.array([params])
        Hs_corr_mat = paramss * Hs_ncorr_mat
        Hs_corr = np.sqrt(np.sum(Hs_corr_mat, axis=1))
        params = np.sqrt(params)
        
        print(' \n ')
        print('Params used for the ' +title.upper()+ ' calibration are: \n ')
        print(params)
        print(' \n ')
        
        # Plotting corrected results
        print('Plotting just the data used to calibrate... \n ')
        
        self.calibration_plots(Hs_ncorr, Hs_corr, hs_calibrate,
                               calibration, params, title)
        
        # Now, we will save all the data corrected
        print('Saving corrected results... \n ')
        
        Hsea    = create_vec_direc(self.hind_to_corr['Hsea_'+self.hind], \
                                   self.hind_to_corr['Dirsea_'+self.hind])
        Hswell1 = create_vec_direc(self.hind_to_corr['Hswell1_'+self.hind], \
                                   self.hind_to_corr['Dirswell1_'+self.hind])
        Hswell2 = create_vec_direc(self.hind_to_corr['Hswell2_'+self.hind], \
                                   self.hind_to_corr['Dirswell2_'+self.hind])
        Hswell3 = create_vec_direc(self.hind_to_corr['Hswell3_'+self.hind], \
                                   self.hind_to_corr['Dirswell3_'+self.hind])
        Hs_ncorr_mat = np.concatenate([Hsea**2, Hswell1**2 + Hswell2**2 + \
                                       Hswell3**2], axis=1)
        Hs_ncorr = np.sqrt(np.sum(Hs_ncorr_mat, axis=1))
        Hs_corr_mat = paramss * Hs_ncorr_mat
        Hs_corr = np.sqrt(np.sum(Hs_corr_mat, axis=1))
        calibration_return = self.hindcast.copy()
        calibration_return.iloc[self.possible_to_correct]\
                               ['Hs_'+self.hind+'_cal'] = Hs_corr
        calibration_return.iloc[self.possible_to_correct]\
                               ['Hs_'+self.hind] = Hs_corr
        print(' \n   \n ')
        
        # return
        return calibration_return, params
    
    
    def satellite_values(self, ini_lat, end_lat, ini_lon, end_lon):
        """ Performs the time calibration step that allows us to perform the
            calibration between the hindcast and the buoy data
            ------------
            Parameters
            Lats and lons to generate the box with the satellite data that
            will be used, previously selected by input
            ------------
            Returns
            Significant height for the satellite and a reduced dataframe for
            the hindcast data
        """
        
        # SATELLITE
        print('Selecting the satellite data choosed... \n ')
        
        self.sat = self.sat.isel(TIME=np.where(self.sat.LATITUDE.values > \
                                               ini_lat)[0])
        self.sat = self.sat.isel(TIME=np.where(self.sat.LATITUDE.values < \
                                               end_lat)[0])
        self.sat = self.sat.isel(TIME=np.where(self.sat.LONGITUDE.values > \
                                               ini_lon)[0])
        self.sat = self.sat.isel(TIME=np.where(self.sat.LONGITUDE.values < \
                                               end_lon)[0])
        
        print('Satellite length: ' + str(len(self.sat.TIME.values)) + ' \n ')
        
        # HINDCAST
        print('Hindcast information able to calibrate: ' + \
              str(len(self.hind_to_corr)) + ' \n ')
        
        # We perform the calibration
        
        print('Choose the way to calibrate the data: ')
        type_calib_way = input('True: hindcast for each satellite \n' + 
                               'False: satellite for each hindcast \n' + 
                               '----- Select ----- : ')
        print(' \n ')
        
        print('Performing the time calibration... \n ')
        times_sat, times_hind = calibration_time(self.sat.TIME.values, 
                                                 self.hind_to_corr.index.values, 
                                                 sh = type_calib_way)
        sat_times = self.sat.sel(TIME=times_sat)
        
        # All the necessary Satellite data (Quality)
        wave_height_qlt = np.nansum(np.concatenate((sat_times['SWH_KU_quality_control'].\
                                                    values[:, np.newaxis], 
                                                    sat_times['SWH_KA_quality_control'].\
                                                    values[:, np.newaxis]), 
                                                    axis = 1), 
                                    axis = 1)
        good_qlt = np.where(wave_height_qlt < 1.5)
        # All necessary Satellite data (Heights)
        wave_height_cal = np.nansum(np.concatenate((sat_times['SWH_KU_CAL'].\
                                                    values[:, np.newaxis], 
                                                    sat_times['SWH_KA_CAL'].\
                                                    values[:, np.newaxis]), 
                                                    axis = 1), 
                                    axis = 1)
        wave_height_cal = wave_height_cal[good_qlt]
        
        calibration = self.hind_to_corr.loc[times_hind].iloc[good_qlt]
        
        print('Length of data to calibrate: ' + str(len(calibration)) + ' \n ')
        
        return wave_height_cal, calibration
    
    
    def calibration_plots(self, xx1, xx2, hs, data, coefs, big_title):
        """ Plots differnet graphs for the calibration of hindcast data with
            buoy and satellite information. Plots are explained by their own
            ------------
            Parameters
            xx1: No corrected data
            xx2: Corrected data
            hs: Satellite / Buoy data that has been used to calibrate
            data: Dataframe with more hindcast information
            coefs: Parameters calculated in the calibration
            big_title: Can be 'Buoy' or 'Satellite'
            ------------
            Returns
            Different auto explicative plots
        """
        
        num='1'
        fig, axs = plt.subplots(2, 3, figsize=(20,20))
        fig.subplots_adjust(hspace=0.2, wspace=0.2)
        fig.suptitle(str(self.hind) + ' hindcast calibration with ' + 
                     big_title + ' data', 
                     fontsize=24, y=0.98, fontweight='bold')
        for i in range(2):
            for j in range(3):
                if (i==j==0 or i==1 and j==0):
                    if i==0:
                        x, y = hs, xx1
                        title = 'No corrected, Hs [m]'
                    else:
                        x, y = hs, xx2
                        title = 'Corrected, Hs [m]'
                        
                    xy = np.vstack([x, y])
                    z = gaussian_kde(xy)(xy) 
                    idx = z.argsort()                                                  
                    x2, y2, z = x[idx], y[idx], z[idx]
                    axs[i,j].scatter(x2, y2, c=z, s=1, cmap=cmocean.cm.haline)
                    axs[i,j].set_xlabel(big_title, fontsize=12, 
                                        fontweight='bold')
                    axs[i,j].set_ylabel('Hindcast', fontsize=12, 
                                        fontweight='bold')
                    axs[i,j].set_title(title, fontsize=12, 
                                       fontweight='bold')
                    maxt = np.ceil(max(max(x)+0.1, max(y)+0.1))
                    axs[i,j].set_xlim(0, maxt)
                    axs[i,j].set_ylim(0, maxt)
                    axs[i,j].plot([0, maxt], [0, maxt], '-k', linewidth=0.6)
                    axs[i,j].set_xticks(np.linspace(0, maxt, 5)) 
                    axs[i,j].set_yticks(np.linspace(0, maxt, 5))
                    axs[i,j].set_aspect('equal')
                    xq = stats.probplot(x2, dist="norm")
                    yq = stats.probplot(y2, dist="norm")
                    axs[i,j].plot(xq[0][1], yq[0][1], "o", markersize=1, 
                                  color='k', label='Q-Q plot')
                    mse = mean_squared_error(x2, y2)
                    rmse_e = rmse(x2, y2)
                    BIAS = bias(x2, y2)
                    SI = si(x2, y2)
                    label = '\n'.join((
                            r'RMSE = %.2f' % (rmse_e, ),
                            r'mse =  %.2f' % (mse,  ),
                            r'BIAS = %.2f' % (BIAS,  ),
                            R'SI = %.2f' % (SI,  )))
                    axs[i,j].text(0.7, 0.05, label, 
                                  transform=axs[i,j].transAxes)
                    
                elif (i==0 and j==1 or i==0 and j==2):
                    if j==1:
                        dataj1 = data[['Dirsea_'+self.hind, \
                                       'Hsea_'+self.hind]].\
                                       dropna(axis=0, how='any')
                        x, y = dataj1['Dirsea_'+self.hind], \
                               dataj1['Hsea_'+self.hind]
                        index = 2
                        title = 'Sea'
                    else:
                        dataj2 = data[['Dirswell'+num+'_'+self.hind, \
                                       'Hswell'+num+'_'+self.hind]].\
                                       dropna(axis=0, how='any')
                        x, y = dataj2['Dirswell'+num+'_'+self.hind], \
                               dataj2['Hswell'+num+'_'+self.hind]
                        index = 3
                        title = 'Swell '+num
                        
                    x = (x*np.pi)/180
                    axs[i,j].axis('off')
                    axs[i,j] = fig.add_subplot(2, 3, index, projection='polar')
                    xy = np.vstack([x, y])
                    z = gaussian_kde(xy)(xy) 
                    idx = z.argsort()                                                  
                    x2, y2, z = x[idx], y[idx], z[idx]
                    axs[i,j].scatter(x2, y2, c=z, s=3, cmap='Blues_r')
                    axs[i,j].set_theta_zero_location('N', offset=0)
                    axs[i,j].set_xticklabels(['N', 'NE', 'E','SE', 
                                              'S', 'SW', 'W', 'NW'])
                    axs[i,j].set_theta_direction(-1)
                    axs[i,j].set_xlabel('Dir [º]', fontsize=12, 
                                        fontweight='bold')
                    axs[i,j].set_ylabel('Hs [m]', labelpad=20, fontsize=12, 
                                        fontweight='bold')
                    axs[i,j].set_title(title, pad=15, fontsize=12, 
                                       fontweight='bold')
                    
                else:
                    if (j==1):
                        color_vals = coefs[0:16]
                        title = 'Sea'
                    else:
                        color_vals = coefs[16:32]
                        title = 'Swell'
                    if np.max(abs(coefs))>1.5:
                        norm = 1
                    else:
                        norm = np.max(coefs)-1
                    fracs = np.repeat(10, 16)
                    my_norm = mpl.colors.Normalize(1-norm, 1+norm)
                    my_cmap = mpl.cm.get_cmap('bwr', len(color_vals))
                    axs[i,j].pie(fracs, labels=None, 
                                 colors=my_cmap(my_norm(color_vals)), 
                                 startangle=90, counterclock=False)
                    axs[i,j].set_title(title, fontsize=12, fontweight='bold')
                    if j==1:
                        ax_cb = fig.add_axes([0.625,0.15,0.02,.25])
                        cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=my_cmap, 
                                                       norm=my_norm)
                        cb.set_label('')
    
    
    def buoy_comparison(self, comparison_type):
        """ Compares data with buoy, even if it's been corrected or not
            ------------
            Parameters
            comparison_type: Type of comparison to be done 
                             (raw, sat_corr, buoy_corr)
            ------------
            Returns
            Different auto explicative plots
        """
        
        # Initialize the data to compare
        if comparison_type=='raw':
            comparison = pd.concat([self.buoy, self.hindcast], axis=1)
        elif comparison_type=='sat_corr':
            comparison = pd.concat([self.buoy, self.hindcast_sat_corr], axis=1)
        elif comparison_type=='buoy_corr':
            comparison = pd.concat([self.buoy, self.hindcast_buoy_corr], axis=1)
        else:
            message = 'Not a valid value for comparison_type'
            return message
        
        print('--------------------------------------------------------')
        print(comparison_type.upper() + ' comparison will be performed')
        print('-------------------------------------------------------- \n ')
        
        comparison = comparison[['Hs_Buoy', 'Tp_Buoy', 'Dir_Buoy',
                                 'Hs_'+self.hind, 
                                 'Tp_'+self.hind, 
                                 'Dir_'+self.hind]]
        
        # Perform the comparison
        n = int(input('Number of years: '))
        years = list(map(int, input('Years separated by one space: ')\
                         .strip().split()))[:n]        
        print(' \n ')
        print('Comparing data... \n ')
        years = [2006, 2007]
        
        for year in years:
            year_plot = comparison.copy()
            ini = str(year)+'-01-01 00:00:00'
            end = str(year)+'-12-31 23:00:00'
            year_plot = year_plot.loc[ini:end]
            fig, axs = plt.subplots(3, 1, figsize=(20,15), sharex=True)
            fig.subplots_adjust(hspace=0.05, wspace=0.1)
            fig.suptitle('Year: ' + str(year) + 
                         ', Bilbao-Vizcaya Ext buoy validation with ' + 
                         comparison_type.upper()+ ' ' +self.hind, 
                         fontsize=24, y=0.94, fontweight='bold')
            months = ['                        Jan', 
                      '                        Feb', 
                      '                        Mar', 
                      '                        Apr', 
                      '                        May', 
                      '                        Jun', 
                      '                        Jul', 
                      '                        Aug', 
                      '                        Sep', 
                      '                        Oct', 
                      '                        Nov', 
                      '                        Dec']
            labels = ['Hs [m]', 'Tp [s]', 'Dir [s]']
            
            i = 0
            while i < 3:
                if i==2:
                    axs[i].plot(year_plot[year_plot.columns.values[i]], '.', 
                                markersize=1, color='darkblue')
                    axs[i].plot(year_plot[year_plot.columns.values[i+3]], '.', 
                                markersize=1, color='red')
                    axs[i].set_ylabel(labels[i], fontsize=12, fontweight='bold')
                    axs[i].grid()
                    axs[i].set_xlim(ini, end)
                    axs[i].set_xticks(np.arange(pd.to_datetime(ini), 
                                                pd.to_datetime(end), 
                                                td(days=30.5)))
                    axs[i].tick_params(direction='in')
                    axs[i].set_xticklabels(months, fontsize=12, 
                                           fontweight='bold')
                else:
                    axs[i].plot(year_plot[year_plot.columns.values[i]], 
                                color='darkblue', linewidth=1)
                    axs[i].plot(year_plot[year_plot.columns.values[i+3]], 
                                color='red', linewidth=1)
                    axs[i].set_ylabel(labels[i], fontsize=12, 
                                      fontweight='bold')
                    axs[i].grid()
                    axs[i].tick_params(direction='in')
                fig.legend(['Buoy', self.hind], loc=(0.66, 0.04), ncol=3, 
                           fontsize=14)
                i += 1
           
     
    def buoy_validation(self, validation_type):
        """ Validate data with buoy, even if it's been corrected or not
            ------------
            Parameters
            validation_type: Type of comparison to be done
                             (raw, sat_corr, buoy_corr)
            ------------
            Returns
            Different auto explicative plots
        """
        
        # Initialize the data to validate
        if validation_type=='raw':
            validation = pd.concat([self.buoy, self.hindcast], axis=1)
            title = 'No previous correction'
        elif validation_type=='sat_corr':
            validation = pd.concat([self.buoy, self.hindcast_sat_corr], axis=1)
            title = 'Previosly corrected with buoy data'
        elif validation_type=='buoy_corr':
            validation = pd.concat([self.buoy, self.hindcast_buoy_corr], axis=1)
            title = 'Previosly corrected with satellite data'
        else:
            message = 'Not a valid value for validation_type'
            return message
        
        print('--------------------------------------------------------')
        print(validation_type.upper() + ' validation will be performed')
        print('-------------------------------------------------------- \n ')
        
        validation = validation[['Hs_Buoy', 'Tp_Buoy', 'Dir_Buoy',
                                 'Hs_'+self.hind, 
                                 'Tp_'+self.hind, 
                                 'Dir_'+self.hind]]
        validation = validation.dropna(axis=0, how='any')
        
        print('Validating and plotting validated data... \n ')
        print('Length of data to validate: ' + str(len(validation)) + ' \n ')
        
        fig, axs = plt.subplots(2, 3, figsize=(20,20))
        fig.subplots_adjust(hspace=0.2, wspace=0.2)
        fig.suptitle('Hindcast: ' + str(self.hind) + 
                     ', Bilbao-Vizcaya Ext buoy validation \n ' +title, 
                     fontsize=24, y=0.98, fontweight='bold')
        
        for i in range(2):
            for j in range(3):
                if (i==j==0 or i==1 and j==0):
                    if i==0:
                        x, y = validation['Hs_Buoy'], \
                               validation['Hs_'+self.hind]
                        title = 'Hs [m]'
                    else:
                        x, y = validation['Tp_Buoy'], \
                               validation['Tp_'+self.hind]
                        title = 'Tp [s]'
                        
                    xy = np.vstack([x, y])
                    z = gaussian_kde(xy)(xy) 
                    idx = z.argsort()                                                  
                    x2, y2, z = x[idx], y[idx], z[idx]
                    axs[i,j].scatter(x2, y2, c=z, s=1, cmap=cmocean.cm.haline)
                    axs[i,j].set_xlabel('Boya', fontsize=12, 
                                        fontweight='bold')
                    axs[i,j].set_ylabel('Modelo', fontsize=12, 
                                        fontweight='bold')
                    axs[i,j].set_title(title, fontsize=12, 
                                       fontweight='bold')
                    maxt = np.ceil(max(max(x)+0.1, max(y)+0.1))
                    axs[i,j].set_xlim(0, maxt)
                    axs[i,j].set_ylim(0, maxt)
                    axs[i,j].plot([0, maxt], [0, maxt], '-k', linewidth=0.6)
                    axs[i,j].set_xticks(np.linspace(0, maxt, 5)) 
                    axs[i,j].set_yticks(np.linspace(0, maxt, 5))
                    axs[i,j].set_aspect('equal')
                    xq = stats.probplot(x2, dist="norm")
                    yq = stats.probplot(y2, dist="norm")
                    axs[i,j].plot(xq[0][1], yq[0][1], "o", markersize=1, 
                                  color='k', label='Q-Q plot')
                    mse = mean_squared_error(x2, y2)
                    rmse_e = rmse(x2, y2)
                    BIAS = bias(x2, y2)
                    SI = si(x2, y2)
                    label = '\n'.join((
                            r'RMSE = %.2f' % (rmse_e, ),
                            r'mse =  %.2f' % (mse,  ),
                            r'BIAS = %.2f' % (BIAS,  ),
                            R'SI = %.2f' % (SI,  )))
                    axs[i,j].text(0.7, 0.05, label, 
                                  transform=axs[i,j].transAxes)
                    
                elif (i==0 and j==1 or i==0 and j==2):
                    idx_buoy = validation['Tp_Buoy'].argsort()
                    idx_hind = validation['Tp_'+self.hind].argsort()
                    if j==1:
                        x, y = validation['Dir_Buoy'][idx_buoy], \
                               validation['Hs_Buoy'][idx_buoy]
                        index = 2
                        c = validation['Tp_Buoy'][idx_buoy]
                        title = 'Boya'
                    else:
                        x, y = validation['Dir_'+self.hind][idx_hind], \
                               validation['Hs_'+self.hind][idx_hind]
                        index = 3
                        c = validation['Tp_'+self.hind][idx_hind]
                        title = 'Modelo'
                    x = (x*np.pi)/180
                    axs[i,j].axis('off')
                    axs[i,j] = fig.add_subplot(2, 3, index, projection='polar')
                    c = axs[i,j].scatter(x, y, c=c, s=5, cmap='magma_r', 
                                         alpha=0.75)
                    cbar = plt.colorbar(c, pad=0.1)
                    cbar.ax.set_ylabel('Tp [s]', fontsize=12, 
                                       fontweight='bold')
                    axs[i,j].set_theta_zero_location('N', offset=0)
                    axs[i,j].set_xticklabels(['N', 'NE', 'E','SE', 
                                              'S', 'SW', 'W', 'NW'])
                    axs[i,j].set_theta_direction(-1)
                    axs[i,j].set_xlabel('Dir [º]', fontsize=12, 
                                        fontweight='bold')
                    axs[i,j].set_ylabel('Hs [m]', labelpad=20, fontsize=12, 
                                        fontweight='bold')
                    axs[i,j].set_title(title, pad=15, fontsize=12, 
                                       fontweight='bold')
                    
                else:
                    if j==1:
                        x, y = validation['Tp_Buoy'], \
                               validation['Hs_Buoy']
                        c = 'darkblue'
                        title = 'Boya'
                    else:
                        x, y = validation['Tp_'+self.hind], \
                               validation['Hs_'+self.hind]
                        c = 'red'
                        title = 'Modelo'
                    xy = np.vstack([x, y])
                    z = gaussian_kde(xy)(xy) 
                    idx = z.argsort()                                                  
                    x2, y2, z = x[idx], y[idx], z[idx]
                    axs[i,j].scatter(x2, y2, c=z, s=3, cmap='Blues_r')
                    axs[i,j].set_xlabel('Tp [s]', fontsize=12, 
                                        fontweight='bold')
                    axs[i,j].set_ylabel('Hs [m]', fontsize=12, 
                                        fontweight='bold')
                    axs[i,j].set_title(title, fontsize=12, 
                                       fontweight='bold')
                    axs[i,j].set_xlim(0, 20)
                    axs[i,j].set_ylim(0, 7.5)
         
            
    def region_map(self):
        """ Maps all the data to a regional zone where buoy, hindcast and
            satellite information can be seen together with the bathymetry.
            This bathymetry will be of great interest for the propagation
            of the calibrated data to shallow waters
            ------------
            Parameters
            ------------
            Returns
            A very good plot though
        """
        
        plt.figure(figsize=(20,20))
        print(' \n' )
        print('MAP BOUNDARYS: ')
        print('--------------------------------------------------------')
        ini_lat = float(input('South latitude: ')) #43.0
        end_lat = float(input('North latitude: ')) #44.4
        ini_lon = float(input('West latitude: '))  #-4.6
        end_lon = float(input('East latitude: '))  #-2.4
        grid_step = float(input('Grid step resolution for the meridians: '))
        depth = float(input('Maximum value for the bathymetry (+): '))
        print('-------------------------------------------------------- \n ')
        
        # GEBCO bathymetry
        print('Plotting gebco bathymetry... \n ')
        self.bath_gebco = self.bath_gebco.sel(lat=slice(ini_lat, end_lat))\
                                         .sel(lon=slice(ini_lon, end_lon)) 
        self.bath_gebco.elevation.plot(vmax=-depth, levels=100, cmap='Blues_r')
        plt.xlabel('')
        plt.ylabel('')
        
        # SPAIN bathymetry
        print('Plotting granulated bathymetry for near coast... \n ')
        bat_reg = np.where((self.bath_spain[:,0]>ini_lon) & 
                           (self.bath_spain[:,0]<end_lon) & 
                           (self.bath_spain[:,1]>ini_lat) & 
                           (self.bath_spain[:,1]<end_lat) & 
                           (self.bath_spain[:,2]<depth))[0]
        smc = plt.scatter(self.bath_spain[:,0][bat_reg], 
                          self.bath_spain[:,1][bat_reg], s=1, 
                          c=-self.bath_spain[:,2][bat_reg], cmap='magma', 
                          vmax=0, vmin=-depth)
        plt.colorbar(smc, extend='both')
        print('La batimetría llega hasta: ' + \
              str(np.min(-self.bath_spain[:,2][bat_reg])))
        
        # Land mask
        m = Basemap(llcrnrlon=ini_lon,  llcrnrlat=ini_lat, 
                    urcrnrlon=end_lon, urcrnrlat=end_lat, 
                    resolution='f')
        m.drawcoastlines()
        m.drawmapboundary()
        m.fillcontinents()
        m.drawmeridians(np.arange(ini_lon, end_lon+grid_step, grid_step), 
                        linewidth=0.5, labels=[1,0,0,1])
        m.drawparallels(np.arange(ini_lat, end_lat+grid_step, grid_step), 
                        linewidth=0.5, labels=[1,0,0,1])
        
        # Satellite used
        sat_lats = self.sat.LATITUDE.values
        sat_lons = self.sat.LONGITUDE.values
        x_sat, y_sat = m(sat_lons, sat_lats)
        x_sat , y_sat = [x-360 for x in list(x_sat)], list(y_sat)
        m.scatter(x_sat, y_sat, marker='o', color='black', 
                  label='Satélite', s=0.01)
        
        # Hindcast used
        print('Coordinates for the hindcast: ')
        x_hind = float(input('Longitude for the hindcast point: '))
        y_hind = float(input('Latitude for the hindcast point: '))
        x_hind, y_hind = m(-3.6, 44.0)
        m.scatter(x_hind, y_hind, marker='o', color='darkred', 
                  label=self.hind, s=75)
        
        # Buoy used
        x_buoy = [-3.05] #float(input('Longitude for the buoy point: '))
        y_buoy = [43.54] #float(input('Latitude for the buoy point: '))
        x_buoy, y_buoy = m(x_buoy, y_buoy)
        m.scatter(x_buoy, y_buoy, marker='D', color='b', 
                  label='Bilbao-Vizcaya', s=75)
        
        plt.title('Mapa de la región seleccionada detallado', pad=20,
                  fontdict={'fontname':'DejaVu Sans', 'size':'18', 'color':'black', 
                            'weight':'normal', 'verticalalignment':'bottom'})
        plt.legend(loc='lower left')
        plt.show()
        
