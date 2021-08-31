[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/javitausia/CalValWaves/master)

# CalValWaves 

### Calibration and validation of hindcast information

CalValWaves is an open-source software toolkit written in python that enables the users to calibrate and validate hindcast data. This hindcast significant wave height data can be calibrated using both buoy and satellite significant height of waves as the "good" measure, but the way we prefer to do it is calibrating first with the satellite data and validating with the buoy after the calibration. A description of what has been done can be seen in this paper:

* João Albuquerque, Jose A. A. Antolínez, Ana Rueda, Fernando J.Méndez, Giovanni Cocoa (November 2018). Directional correction of modeled sea and swell wave heights using satellite altimeter data. https://doi.org/10.1016/j.ocemod.2018.09.001

## 1. Description

Numerical reanalysis are very useful as they have information of different variables for a very long and constant period of time. In this case, this data will be used after been calibrated to propagate waves from the point where the node of the reanalysis is located to shallow waters, and it is very important to have constant data along a very large period of time, so the propagations can be representative. The problem now is that these hindcasts are not perfect, and this is why we must calibrate them first with satellite data and validate them after all with a nerby buoy to see if data correlates.

The initial area of study is the north of Spain (an explanation map can be seen below) although more locations might be added.

![map](/images/mapa-resumen.png)

In this map, the node of reanalysis is shown with a red mark, the different satellite measures used to calibrate are represented with black points and the final validation will be done using the data of the buoy in purple. The calibration results are shown in this figure:

![calibration](/images/calibration-satellite.png)

For the validation, two figures will be shown. The first compares historically the performance of the buoy and the hindcast, once the significant wave height has been corrected with the satellite, the second one shows different representative characteristics of both buoy and hindcast data:

![comparison3](/images/comparison-satcorr-2008.png)
![validation](/images/validation-satellite.png)

## 2. Data download

Example data is proportioned to the user to see how the toolbox works but in case the process needs to be donde in another location along the world, then this data will have to be acquired and preprocessed for that new place. There is not a unique way to obtain the data, but here is how we have donde it.

- Buoy: This is the most relative part, as it depends on the country you are working on, for Spain, as it is our area of study, data can be requested using the goverment resources: http://www.puertos.es/es-es/oceanografia/Paginas/portus.aspx

- Hincast: Different hindcasts are open to users. In this work, we have used both CSIRO and ERA5, but we have finally decided CSIRO is the most suitable for the purpose of the global project. Both hidcasts can be downloaded online. For the CSIRO hindcast, the user is redirected [here](data-cbr.csiro.au/thredds/ncss/grid/catch_all/CMAR_CAWCR-Wave_archive/CAWCR_Wave_Hindcast_aggregate/gridded/ww3.pac_4m.202107.nc/dataset.html).

- Satellite: IMOS satellite data has been used and it can be downloaded from the AODN (Australian Ocean Data Network) website: https://portal.aodn.org.au/. The next figure explains the steps to follow to correctly dowload the data, be very careful with the option selected, as it must be the one shown in the photo below, which is "IMOS - SRS Surface Waves Sub-Facility - altimeter wave/wind" that could appear not in the first page of the website.

![satellite-steps](/data/satellite/steps.png)

After clicking donwload .txt file and checking everything is correct, you can download the netCDFs files using `cd` to move to the folder where you want to store the data and then running the command `wget - i IMOS... .txt`. Finally, join the files using [concat_satellite_files.py](/data/satellite/concat_satellite_files.py). Another option is to download the netCDF files directly from the IMOS website, then, as before, copy and paste the downloaded folder in `data/satellite` and concat the files!!

### More detailed information could be updated regarding the acquisition of the data (buoy and wave reanalysis (hindcast)).

## 3. Main contents

[calval](./calval/): Python basic files 
- [calval-py](./calval/calval.py): Autocontent class with calibration and validation tools
- [functions](./calval/functions.py): Useful functions used

[tests](./scripts/): Test examples
- [python example](./scripts/example_01.py): Example of how to use the library

[notebooks](./notebooks/): Notebook examples
- [cantabria example](./notebooks/example_jupyter_can.ipynb): Example of CalValWaves in in the north of Spain
- [oahu example](./notebooks/example_jupyter_oahu.ipynb): Example of CalValWaves in in the north of Oahu (Hawaii)

[images](./images/): Image examples
- All the images in this folder can be obtained using the python example. It is not necessary to explain what each image contains as they are self-explicative

[data](./data/): Data used
- All the data present in this folder is enough to run the python files and the jupyter notebook as a first example. If the toolbox wanna be used to calibrate and validate different data, it is compulsary to have a look in how the initial data has been preprocessed. For the hindcast and the buoy data, pandas dataframes / netcdf datasets are used, for the satellite, a netCDF file (as it is downloaded from the IMOS website)

[hindcast files](./data/hindcast/): Hindcast code files
- These files help the user preprocess the netcdf hindcast file downloaded from csiro website, so the python example script and the jupyter notebook can be run easily

[satellite files](./data/satellite/): Satellite code files
- These files help the user join all the sub netcdf files downloaded from the .txt initial file, so the python example script and the jupyter notebook can be run easily

## 4. Installation

### 4.1 Create an environment in conda

To run the toolbox you first need to install the required Python packages in an environment. To do this we will see **Anaconda**, which can be downloaded freely [here](https://www.anaconda.com/download/). Moreover, **miniconda** is also highly recommended, which can be downloaded [here](https://docs.conda.io/en/latest/miniconda.html).

Once you have installed it on your PC, open the Anaconda prompt (in Mac and Linux, open a terminal window) and use the `cd` command to go to the folder where you have cloned this repository.

Create a new environment named `calval` with all the required packages:

```
conda env create -f environment.yml
```

### 4.2 Activate conda environment

All the required packages have been now installed in an environment called `calval`. Now, activate this new environment:

```
conda activate calval
```

## 5. Play

Now everything has been installed, you can now start to play with the python code and the jupyter notebook explanation. Be careful, as some important parameters can be adjusted during the calibration process (construction of the object of the class, first line code in the jupyter notebook). Nevertheless, parameters used are also shown in the example.

Areas available to play are:

![areasplay](/images/areasplay.png)

## Additional support:

Data used in the project and a detailed explanation of the acquisition can be requested from jtausiahoyal@gmail.com or tausiaj@unican.es.

## Author:

* Javier Tausía Hoyal
