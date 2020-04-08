# CalValWaves 

### Calibration and validation of hindcast information

CalValWaves is a open-source software toolkit written in python that enables the users to calibrate and validate hindcast data. This hindcast significant wave height data can be calibrated using both buoy and satellite significant height of waves as the "good" measure, but the way we prefer to do it is calibrating first with the satellite data and validating with the buoy after the calibration. A description of what has been done can be seen in this paper:

* João Albuquerque, Jose A. A. Antolínez, Ana Rueda, Fernando J.Méndez, Giovanni Cocoa (November 2018). Directional correction of modeled sea and swell wave heights using satellite altimeter data. https://doi.org/10.1016/j.ocemod.2018.09.001

## 1. Description

Hindcast models are very useful as they have information of different variables for a very long and constant period of time. In this case, this data will be used after been calibrated to propagate waves from the point where the node of the hindcast is located to shallow waters, and it is very important to have constant data along a very large period of time, so the propagations can be representative. The problem now is that these hindcasts are sometimes a little incorrect, and this is why we must calibrate them first with satellite data and validate them after all with a nerby buoy to see if data correlates.

The area of study is the north of Spain (an explanation map can be seen below). 

![map](/images/mapa-resumen.png)

In this map, the node of reanalysis is shown with a red mark, the different satellite measures used to calibrate are represented with black points and the final validation will be done using the data of the buoy in purple. The calibration results can be shown in this figure:

![calibration](/images/calibration-satellite.png)

For the validation, two figures will be shown. The first compares historically the performance of the buoy and the hindcast, once the significant wave height has been corrected with the satellite, the second one shows different representative characteristics of both buoy and hindcast data:

![comparison](/images/comparison-satcorr-2007.png)
![validation](/images/validation-satellite.png)

## 2. Data download

Example data is proportioned to the user to see how the toolbox works but in case the process needs to be donde in another location along the world, then this data will have to be acquired and preprocessed for that new place. There is not a unique way to obtain the data, but here is how we have donde it.

- Buoy: This is the most relative part, as it depends of the country you are working on, for Spain, as it is our area of study, data can be requested using the goverment resources. http://www.puertos.es/es-es/oceanografia/Paginas/portus.aspx

- Hincast: Different hindcasts are open to users. In this work, we have used both CSIRO and ERA5, but we have finally decided CSIRO is the most suitable for the purpose of the global project. Both hidcasts can be requested online.

- Satellite: IMOS satellite data has been used and it can be downloaded from the AODN (Australian Ocean Data Network) website. https://portal.aodn.org.au/ . The next figure explains the steps to follow to correctly dowload the data:

![satdata](/data/satellite/steps.png)

After clicking donwload .txt file and checking everything is correct, you can download the netCDFs files using `cd` to move to the folder where you want to store the data and then running the command `wget - i IMOS... .txt`. Finally, join the files using [join satellite](/data/satellite/extract_satellite.py).

### More detailed information could be updated regarding the acquisition of the data.

## 3. Main contents

[lib](./lib/): Python basic files 
- [calval](./lib/calval.py): Autocontent class with calibration and validation tools
- [functions](./lib/functions.py): Useful functions used

[test](./tests/): Test examples
- [python example](./tests/example_01.py): Example of how to use the library

[images](./images/): Image examples
- All the images present in this folder can be obtained using the python example. It is not necessary to explain what each image contains as they are self-explicative

[data](./data/): Data used
- All the data present in this folder is enough to run the python files and the jupyter notebook as a first example. If the toolbox wanna be used to calibrate and validate different data, it is compulsary to have a look in how the initial data has been preprocessed. For the hindcast and the buoy data, pandas dataframes are used, for the satellite, a netCDF file (as it is downloaded from the IMOS website).

## 4. Installation

### 4.1 Create an environment in conda

To run the toolbox you first need to install the required Python packages in an environment. To do this we will see **Anaconda**, which can be downloaded freely [here](https://www.anaconda.com/download/).

Once you have installed it on your PC, open the Anaconda prompt (in Mac and Linux, open a terminal window) and use the `cd` command to go to the folder where you have cloned this repository.

Create a new environment named `calval` with all the required packages:

```
conda env create -f environment.yml -n calval
```
### 4.2 Activate conda environment

All the required packages have been now installed in an environment called `calval`. Now, activate this new environment:

```
conda activate calval
```

## 5. Play

Now everything has been installed, you can now start to play with the python code and the jupyter notebook explanation. Be careful, as some important parameters can be adjusted during the calibration process (construction of the object of the class, first line code in the jupyter notebook). Nevertheless, parameters used are also shown in the example.

## Additional support:

Data used in the project and a detailed explanation of the acquisition can be requested from jtausiahoyal@gmail.com.

## Author:

* Javier Tausía Hoyal
