import numpy as np
import pandas as pd
import xarray as xr


print('Creating and saving the dataframe...')

csiro_dataset = xr.open_dataset('csiro_dataset_oahu.nc')

csiro_sel = csiro_dataset.isel(latitude=0,longitude=0)

csiro = pd.DataFrame({'Hs':         csiro_sel.hs.values,
                      'Tm_01':      csiro_sel.t.values,
                      'Tm_02':      csiro_sel.t02.values,
                      'Tp':         1/csiro_sel.fp.values,
                      'DirM':       csiro_sel.dir.values,
                      'DirP':       csiro_sel.dp.values,
                      'Spr':        csiro_sel.spr.values,
                      'Nwp':        csiro_sel.pnr.values,
                      'U10':        csiro_sel.U10.values,
                      'V10':        csiro_sel.V10.values,
                      'Hsea':       csiro_sel.hs0.values,
                      'Hswell1':    csiro_sel.hs1.values,
                      'Hswell2':    csiro_sel.hs2.values,
                      'Hswell3':    csiro_sel.hs3.values,
                      'Tpsea':      csiro_sel.tp0.values,
                      'Tpswell1':   csiro_sel.tp1.values,
                      'Tpswell2':   csiro_sel.tp2.values,
                      'Tpswell3':   csiro_sel.tp3.values,
                      'Dirsea':     csiro_sel.th0.values,
                      'Dirswell1':  csiro_sel.th1.values,
                      'Dirswell2':  csiro_sel.th2.values,
                      'Dirswell3':  csiro_sel.th3.values,
                      'Sprsea':     csiro_sel.si0.values,
                      'Sprswell1':  csiro_sel.si1.values,
                      'Sprswell2':  csiro_sel.si2.values,
                      'Sprswell3':  csiro_sel.si3.values},
                     index = csiro_sel.time.values)

csiro['Hsea'].iloc[np.isnan(csiro['Hsea'].values)] = 0.0
csiro['Hswell1'].iloc[np.isnan(csiro['Hswell1'].values)] = 0.0
csiro['Hswell2'].iloc[np.isnan(csiro['Hswell2'].values)] = 0.0
csiro['Hswell3'].iloc[np.isnan(csiro['Hswell3'].values)] = 0.0
csiro['Hs_cal'] = np.sqrt(csiro['Hsea']**2 + csiro['Hswell1']**2 + csiro['Hswell2']**2 + csiro['Hswell3']**2)
csiro.index = csiro.index.round('H')

# WIND additional COMPONENTS
WIND = np.zeros(len(csiro))
DIRW = np.zeros(len(csiro))
for w in range(len(csiro)):
    h = csiro['U10'].iloc[w]
    v = csiro['V10'].iloc[w]
    if (h>=0 and v>=0):
        h = abs(h)
        v = abs(v)
        wind = np.sqrt(h**2 + v**2)
        dirW = np.arcsin(v/wind)*180/np.pi
        WIND[w] = wind
        DIRW[w] = 90 - dirW
    elif (h>=0 and v<=0):
        h = abs(h)
        v = abs(v)
        wind = np.sqrt(h**2 + v**2)
        dirW = np.arcsin(v/wind)*180/np.pi
        WIND[w] = wind
        DIRW[w] = 90 + dirW
    elif (h<=0 and v<=0):
        h = abs(h)
        v = abs(v)
        wind = np.sqrt(h**2 + v**2)
        dirW = np.arcsin(v/wind)*180/np.pi
        WIND[w] = wind
        DIRW[w] = 180 + (90 - dirW)
    elif (h<=0 and v>=0):
        h = abs(h)
        v = abs(v)
        wind = np.sqrt(h**2 + v**2)
        dirW = np.arcsin(v/wind)*180/np.pi
        WIND[w] = wind
        DIRW[w] = 270 + dirW
    else:
        WIND[w] = np.nan
        DIRW[w] = np.nan
DIRW = DIRW + 180
DIRW[np.where(DIRW>360)[0]] = DIRW[np.where(DIRW>360)[0]] - 360

csiro.insert(10, 'W', WIND)
csiro.insert(11, 'DirW', DIRW)

csiro.to_pickle('csiro_dataframe_tonga.pkl')
csiro.to_xarray().to_netcdf('csiro_dataset_tonga.nc')


