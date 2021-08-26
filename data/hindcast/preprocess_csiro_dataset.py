import numpy as np
import pandas as pd
import xarray as xr


print('Creating and saving the dataframe...')

csiro_dataset = xr.open_dataset('gridded_area_oahu.nc')

csiro_sel = csiro_dataset.isel(latitude=3,longitude=1)
csiro_attrs = csiro_sel.attrs # save attributes

csiro_sel = xr.merge([
    csiro_sel.hs, csiro_sel.t, csiro_sel.t02, csiro_sel.fp, 
    csiro_sel.dir, csiro_sel.dp, csiro_sel.spr, csiro_sel.pnr,
    xr.concat([csiro_sel.U10,csiro_sel.uwnd],dim='time').dropna(dim='time'),
    xr.concat([csiro_sel.V10,csiro_sel.vwnd],dim='time').dropna(dim='time'),
    xr.concat([csiro_sel.hs0,csiro_sel.phs0],dim='time').dropna(dim='time'),
    xr.concat([csiro_sel.hs1,csiro_sel.phs1],dim='time').dropna(dim='time'),
    xr.concat([csiro_sel.hs2,csiro_sel.phs2],dim='time').dropna(dim='time'),
    xr.concat([csiro_sel.hs3,csiro_sel.phs3],dim='time').dropna(dim='time'),
    xr.concat([csiro_sel.tp0,csiro_sel.ptp0],dim='time').dropna(dim='time'),
    xr.concat([csiro_sel.tp1,csiro_sel.ptp1],dim='time').dropna(dim='time'),
    xr.concat([csiro_sel.tp2,csiro_sel.ptp2],dim='time').dropna(dim='time'),
    xr.concat([csiro_sel.tp3,csiro_sel.ptp3],dim='time').dropna(dim='time'),
    xr.concat([csiro_sel.th0,csiro_sel.pdir0],dim='time').dropna(dim='time'),
    xr.concat([csiro_sel.th1,csiro_sel.pdir1],dim='time').dropna(dim='time'),
    xr.concat([csiro_sel.th2,csiro_sel.pdir2],dim='time').dropna(dim='time'),
    xr.concat([csiro_sel.th3,csiro_sel.pdir3],dim='time').dropna(dim='time'),
    xr.concat([csiro_sel.si0,csiro_sel.pspr0],dim='time').dropna(dim='time'),
    xr.concat([csiro_sel.si1,csiro_sel.pspr1],dim='time').dropna(dim='time'),
    xr.concat([csiro_sel.si2,csiro_sel.pspr2],dim='time').dropna(dim='time'),
    xr.concat([csiro_sel.si3,csiro_sel.pspr3],dim='time').dropna(dim='time'),
])
csiro_sel.attrs = csiro_attrs

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

csiro.to_pickle('csiro_dataframe_oahu.pkl')
csiro.to_xarray().assign_attrs(csiro_attrs).to_netcdf('csiro_dataset_oahu.nc')


