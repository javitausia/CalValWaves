# basic
import numpy as np
import pandas as pd

# plotting
import matplotlib as mpl
from matplotlib import pyplot as plt
import cmocean
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# additional
from datetime import timedelta as td

# additional*
import scipy.stats as stats
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import statsmodels.api as sm

# custom
from .functions import rmse, bias, si, create_vec_direc, calibration_time


# Calibration-validation class
class CalVal(object):
    
    
    def __init__(self, buoy, hindcast, sat):
        """ Initializes the class with all the necessary attributes that
            will be used in the different methods
            ------------
            Parameters
            buoy: Buoy data as a dataframe
            hindcast: Hindcast data as a dataframe
            sat: Satellite data as a netCDF (see extract_satellite.py)
            ------------
            Returns
            Attributes initialized and plots for both calibrations, the one
            done with the buoy and the other one performed with satellite
            data. The parameters for the calibrations are also stored
        """
                
        self.buoy                 =    buoy
        self.hindcast             =    hindcast.copy()
        self.possible_to_correct  =    np.where(hindcast['Hs_cal'] > 0.01)[0]        
        self.hind_to_corr         =    hindcast.iloc[self.possible_to_correct]
        self.sat                  =    sat
        self.hindcast_sat_corr    =    hindcast.copy()
        self.params_sat_corr      =    self.calibration('sat')
        self.hindcast_buoy_corr   =    hindcast.copy()
        self.params_buoy_corr     =    self.calibration('buoy')
    
    
    def calibration(self, calibration_type):
        """ Calibrates hindcast with satellite or buoy data. This calibration
            is performed using a linear regression and selecting only those
            parameters that are significantly representative
            ------------
            Parameters
            calibration_type: (Str) Type of calibration to be done (sat, buoy)
            ------------
            Returns
            Corrected data and calculated params
        """
        
        print('--------------------------------------------------------')
        print(calibration_type.upper() + ' CALIBRATION will be performed')
        print('-------------------------------------------------------- \n ')
        
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
            notna_values = calibration_tot['Hs_Buoy'].notna().values
            calibration = calibration_tot.iloc[notna_values]
            hs_calibrate = calibration['Hs_Buoy'].values
            title = 'Buoy'
        else:
            return 'Not a valid value for calibration_type'
        
        print(' \n ')
        
        # Construct matrices to calibrate
        print('Constructing matrices and calibrating... \n ')
        
        Hsea    = create_vec_direc(calibration['Hsea'], \
                                   calibration['Dirsea'])
        Hswell1 = create_vec_direc(calibration['Hswell1'], \
                                   calibration['Dirswell1'])
        Hswell2 = create_vec_direc(calibration['Hswell2'], \
                                   calibration['Dirswell2'])
        Hswell3 = create_vec_direc(calibration['Hswell3'], \
                                   calibration['Dirswell3'])
        Hs_ncorr_mat = np.concatenate([Hsea**2, Hswell1**2 + \
                                       Hswell2**2 + Hswell3**2], 
                                      axis=1)
        Hs_ncorr = np.sqrt(np.sum(Hs_ncorr_mat, axis=1))
        #---------------------------------------------------------------------#
        print('Value to set the umbral for not enough data to calibrate, ')
        print('this value can be from 0.01 to 0.03: ' )
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
        
        Hsea    = create_vec_direc(self.hind_to_corr['Hsea'], \
                                   self.hind_to_corr['Dirsea'])
        Hswell1 = create_vec_direc(self.hind_to_corr['Hswell1'], \
                                   self.hind_to_corr['Dirswell1'])
        Hswell2 = create_vec_direc(self.hind_to_corr['Hswell2'], \
                                   self.hind_to_corr['Dirswell2'])
        Hswell3 = create_vec_direc(self.hind_to_corr['Hswell3'], \
                                   self.hind_to_corr['Dirswell3'])
        Hsea_corr_mat    = paramss[:,0:16]  * Hsea**2
        Hswell1_corr_mat = paramss[:,16:32] * Hswell1**2
        Hswell2_corr_mat = paramss[:,16:32] * Hswell2**2
        Hswell3_corr_mat = paramss[:,16:32] * Hswell3**2
        Hsea_corr        = np.sqrt(np.sum(Hsea_corr_mat, axis=1))
        Hswell1_corr     = np.sqrt(np.sum(Hswell1_corr_mat, axis=1))
        Hswell2_corr     = np.sqrt(np.sum(Hswell2_corr_mat, axis=1))
        Hswell3_corr     = np.sqrt(np.sum(Hswell3_corr_mat, axis=1))
        Hs_ncorr_mat     = np.concatenate([Hsea**2, Hswell1**2 + Hswell2**2 + \
                                           Hswell3**2], axis=1)
        Hs_ncorr         = np.sqrt(np.sum(Hs_ncorr_mat, axis=1))
        Hs_corr_mat      = paramss * Hs_ncorr_mat
        Hs_corr          = np.sqrt(np.sum(Hs_corr_mat, axis=1))
        index_hs      = np.where(self.hindcast.columns.values=='Hs')[0][0]
        index_hsea    = np.where(self.hindcast.columns.values=='Hsea')[0][0]
        index_hswell1 = np.where(self.hindcast.columns.values=='Hswell1')[0][0]
        index_hswell2 = np.where(self.hindcast.columns.values=='Hswell2')[0][0]
        index_hswell3 = np.where(self.hindcast.columns.values=='Hswell3')[0][0]
        index_hs_cal  = np.where(self.hindcast.columns.values=='Hs_cal')[0][0]
        
        if calibration_type=='sat':
            self.hindcast_sat_corr.iloc[self.possible_to_correct, index_hs] = Hs_corr
            self.hindcast_sat_corr.iloc[self.possible_to_correct, index_hsea] = Hsea_corr
            self.hindcast_sat_corr.iloc[self.possible_to_correct, index_hswell1] = Hswell1_corr
            self.hindcast_sat_corr.iloc[self.possible_to_correct, index_hswell2] = Hswell2_corr
            self.hindcast_sat_corr.iloc[self.possible_to_correct, index_hswell3] = Hswell3_corr
            self.hindcast_sat_corr.iloc[self.possible_to_correct, index_hs_cal] = Hs_corr
        else:
            self.hindcast_buoy_corr.iloc[self.possible_to_correct, index_hs] = Hs_corr
            self.hindcast_buoy_corr.iloc[self.possible_to_correct, index_hsea] = Hsea_corr
            self.hindcast_buoy_corr.iloc[self.possible_to_correct, index_hswell1] = Hswell1_corr
            self.hindcast_buoy_corr.iloc[self.possible_to_correct, index_hswell2] = Hswell2_corr
            self.hindcast_buoy_corr.iloc[self.possible_to_correct, index_hswell3] = Hswell3_corr
            self.hindcast_buoy_corr.iloc[self.possible_to_correct, index_hs_cal] = Hs_corr
        print(' \n  \n ')
        
        # return
        return params
    
    
    def satellite_values(self, ini_lat, end_lat, ini_lon, end_lon):
        """ Performs the time calibration step that allows us to perform the
            calibration between the hindcast and the satellite data
            ------------
            Parameters
            Lats and lons to generate the box with the satellite data that
            will be used, previously selected by input
            ------------
            Returns
            Significant wave height for the satellite and a reduced dataframe
            for the hindcast data
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
        fig.suptitle('CSIRO hindcast calibration with ' + 
                     big_title + ' data', 
                     fontsize=24, y=0.98, fontweight='bold')
        for i in range(2):
            for j in range(3):
                if (i==j==0 or i==1 and j==0):
                    if i==0:
                        x, y = hs, xx1
                        title = 'No corrected, $H_S$ [m]'
                    else:
                        x, y = hs, xx2
                        title = 'Corrected, $H_S$ [m]'
                        
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
                        dataj1 = data[['Dirsea', \
                                       'Hsea']].\
                                       dropna(axis=0, how='any')
                        x, y = dataj1['Dirsea'], \
                               dataj1['Hsea']
                        index = 2
                        title = 'Sea'
                    else:
                        dataj2 = data[['Dirswell'+num, \
                                       'Hswell'+num]].\
                                       dropna(axis=0, how='any')
                        x, y = dataj2['Dirswell'+num], \
                               dataj2['Hswell'+num]
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
                    axs[i,j].set_xlabel('$\u03B8_{m}$ [$\degree$]', fontsize=12, 
                                        fontweight='bold')
                    axs[i,j].set_ylabel('$H_S$ [m]', labelpad=20, fontsize=12, 
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
                    #if np.max(abs(coefs))>1.5:
                    #    norm = 1
                    #else:
                    #    norm = np.max(coefs)-1
                    norm = 0.3
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
            return 'Not a valid value for comparison_type'
        
        print('--------------------------------------------------------')
        print(comparison_type.upper() + ' comparison will be performed')
        print('-------------------------------------------------------- \n ')
        
        comparison = comparison[['Hs_Buoy', 'Tp_Buoy', 'Dir_Buoy',
                                 'Hs', 'Tp', 'DirM']]
        
        # Perform the comparison
        n = int(input('Number of years: '))
        years = list(map(int, input('Years separated by one space: ')\
                         .strip().split()))[:n]
        # years = [2006, 2007, 2008]
        
        print(' \n ')
        print('Comparing data... \n ')
        
        for year in years:
            year_plot = comparison.copy()
            ini = str(year)+'-01-01 00:00:00'
            end = str(year)+'-12-31 23:00:00'
            year_plot = year_plot.loc[ini:end]
            fig, axs = plt.subplots(3, 1, figsize=(20,15), sharex=True)
            fig.subplots_adjust(hspace=0.05, wspace=0.1)
            fig.suptitle('Year: ' + str(year) + 
                         ', Bilbao-Vizcaya Ext buoy validation with ' + 
                         comparison_type.upper()+ ' CSIRO', 
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
            labels = ['$H_S$ [m]', '$T_P$ [s]', '$\u03B8_{m}$ [$\degree$]']
            
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
                fig.legend(['Buoy', 'Modelo'], loc=(0.66, 0.04), ncol=3, 
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
            return 'Not a valid value for validation_type'
        
        print('--------------------------------------------------------')
        print(validation_type.upper() + ' VALIDATION will be performed')
        print('-------------------------------------------------------- \n ')
        
        validation = validation[['Hs_Buoy', 'Tp_Buoy', 'Dir_Buoy',
                                 'Hs', 'Tp', 'DirM']]
        validation = validation.dropna(axis=0, how='any')
        
        print('Validating and plotting validated data... \n ')
        print('Length of data to validate: ' + str(len(validation)) + ' \n ')
        
        fig, axs = plt.subplots(2, 3, figsize=(20,20))
        fig.subplots_adjust(hspace=0.2, wspace=0.2)
        fig.suptitle('Hindcast: CSIRO' + 
                     ', Bilbao-Vizcaya Ext buoy validation \n ' +title, 
                     fontsize=24, y=0.98, fontweight='bold')
        
        for i in range(2):
            for j in range(3):
                if (i==j==0 or i==1 and j==0):
                    if i==0:
                        x, y = validation['Hs_Buoy'], \
                               validation['Hs']
                        title = '$H_S$ [m]'
                    else:
                        x, y = validation['Tp_Buoy'], \
                               validation['Tp']
                        title = '$T_P$ [s]'
                        
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
                    idx_hind = validation['Tp'].argsort()
                    if j==1:
                        x, y = validation['Dir_Buoy'][idx_buoy], \
                               validation['Hs_Buoy'][idx_buoy]
                        index = 2
                        c = validation['Tp_Buoy'][idx_buoy]
                        title = 'Boya'
                    else:
                        x, y = validation['DirM'][idx_hind], \
                               validation['Hs'][idx_hind]
                        index = 3
                        c = validation['Tp'][idx_hind]
                        title = 'Modelo'
                    x = (x*np.pi)/180
                    axs[i,j].axis('off')
                    axs[i,j] = fig.add_subplot(2, 3, index, projection='polar')
                    c = axs[i,j].scatter(x, y, c=c, s=5, cmap='magma_r', 
                                         alpha=0.75)
                    cbar = plt.colorbar(c, pad=0.1)
                    cbar.ax.set_ylabel('$T_P$ [s]', fontsize=12, 
                                       fontweight='bold')
                    axs[i,j].set_theta_zero_location('N', offset=0)
                    axs[i,j].set_xticklabels(['N', 'NE', 'E','SE', 
                                              'S', 'SW', 'W', 'NW'])
                    axs[i,j].set_theta_direction(-1)
                    axs[i,j].set_xlabel('$\u03B8_{m}$ [$\degree$]', fontsize=12, 
                                        fontweight='bold')
                    axs[i,j].set_ylabel('$H_S$ [m]', labelpad=20, fontsize=12, 
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
                        x, y = validation['Tp'], \
                               validation['Hs']
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

