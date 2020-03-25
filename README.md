# CalValWaves 

### Calibration and validation of hindcast information

CalValWaves is a open-source software toolkit written in python that enables the users to calibrate and validate hindcast data. This hindcast significant height data can be calibrated using both buoy and satellite significant height of waves as the "good" measure, but the way we prefer to do it is calibrating first with he satellite data and validating the with the buoy. A description of what has been done can be seen in this paper:

* João Albuquerque, Jose A. A. Antolínez, Ana Rueda, Fernando J.Méndez, Giovanni Cocoa (November 2018). Directional correction of modeled sea and swell wave heights using satellite altimeter data. https://doi.org/10.1016/j.ocemod.2018.09.001

## 1. Description

Hindcast models are very useful as they have information of different variables for a very long and constant period of time. In this case, this data will be used after been calibrated to propagate waves from the point where the node of the hindcast is to shallow waters, and it is very important to have constant data along a very large period of time, so the propagations can be representative. The problem now is that these hindcasts are sometimes wrong, and this is why we must calibrate them first with satellite data and validate them after all with a nerby buoy to see if data correlate.

Example data is proportioned to the user to see how the toolbox works but in case the process needs to be donde in another location along the world, then this data will have to be acquired and preprocessed for that new place. There is not a unique way to obtain the data, but here is how we have donde it.

- Buoy: This is the most relative part, as it depends of the country you are working on, for Spain, as it is our area of study, data can be requested using the goverment resources.
- Hincast: Different hindcasts are open to users. In this work, we have used both CSIRO and ERA5, but we have finally decided CSIRO is most suitable for the purpose of the global project. Both hidcasts can be requested online.
- Satellite: IMOS satellite data has been used and it can be downloaded for the IMOS website.
- Bathymetry: The GEBCO bathymetry is constant along the world and it can be downloaded as a netCDF file along thw world. HOwever, depending on the purpose of the project, more precise data should be requested. Here we use more precise bathymetry data in Spain.

## 2. Main contents

[lib](./lib/): Python basic files 
- [calval](./lib/calval.py): Class autocontent with calibration and validation tools
- [functions](./lib/functions.py): Useful functions used

[test](./tests/): Test examples
- [python example](./tests/example_01.py): Example of how to use the library
- [notebook example](./test/example_01.ipynb): Jupyter notebook explanation.

## 3. Installation

### 3.1 Create an environment in conda

To run the toolbox you first need to install the required Python packages in an environment. To do this we will see **Anaconda**, which can be downloaded freely [here](https://www.anaconda.com/download/).

Once you have installed it on your PC, open the Anaconda prompt (in Mac and Linux, open a terminal window) and use the `cd` command to go to the folder where you have cloned this repository.

Create a new environment named `calval` with all the required packages:

```
conda env create -f environment.yml -n calval
```

All the required packages have been now installed in an environment called `calval`. Now, activate this new environment:

```
conda activate calval
```

### 3.2 Play

Now everything has been installed, you can now start to play with the python code and the jupyter notebook explanation.

## Additional support:

Data used in the project and a detailed explanation of the acquisition can be requested from jtausiahoyal@gmail.com.

## Author:

* Javier Tausía Hoyal
