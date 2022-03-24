# Data storage

In this `CalValWaves/data` sub-directory, all the data will be saved. For this calibration and validation method, we have two available examples with data for Hawaii (Oahu island) and also data for a location in front of Santander (Cantabria, CAN, Spain). For each location, we downloaded the CSIRO hindcast, the satellite data available around the area from the IMOS website and also buoy data if available. The structure of the data is then as follows:

* [hindcast](./hindcast/): This is the historic data from the [CSIRO](data-cbr.csiro.au/thredds/ncss/grid/catch_all/CMAR_CAWCR-Wave_archive/CAWCR_Wave_Hindcast_aggregate/gridded/ww3.pac_4m.202107.nc/dataset.html) website for both example locations (Oahu and Cantabria).

* [satellite](./satellite/): This is the altimeter data from the [IMOS](https://portal.aodn.org.au/) website. Both example sites are also included here, and please have a look at the repository [README](https://github.com/javitausia/CalValWaves#2-data-download) file, so this data download step is explained in detail.

* [buoy](./buoy/): This is buoy information, which is always the ground truth, and used to be compared with every result we obtain and we want to validate. Here, we download the Santander buoy data from the national website [here](http://www.puertos.es/es-es/oceanografia/Paginas/portus.aspx), and we also downloaded the hawaiian buoy data from the [NOAA website](https://www.ndbc.noaa.gov/), where much more buoys around the world can be found.

for additional information regarding both the data and its acquisition, please contact me at jtausiahoyal@gmail.com or tausiaj@unican.es (preferred)!!