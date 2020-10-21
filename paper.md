---
title: 'CalValWaves: A Python package for wave reanalysis calibration with satellite altimeter data'
tags:
  - Python
  - wave reanalysis
  - calibration
  - satellite data
  - buoy validation
authors:
  - name: Javuer Tausia Hoyal^[Custom footnotes for e.g. denoting who the corresponding author is can be included like this.]
    orcid: 0000-0002-2299-2915
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Fernando J. Méndez Incera
    affiliation: "1, 2"
affiliations:
 - name: University of Cantabria
   index: 1
 - name: Geocean group, ETSICCP
   index: 2
date: 7 October 2020
bibliography: paper.bib
---

# Summary

Numerical wave reanalysis are very useful as they have information of different 
variables for a very long and constant period of time. In this case, this data 
will be used after been calibrated to propagate waves from the point where the 
node of the reanalysis is located to shallow waters, and it is very important 
to have constant data along a very large period of time, so the propagations can 
be representative. The problem now is that these hindcasts are not perfect, and 
this is why we must calibrate them first with satellite data and validate them after 
all with a nerby buoy to see if data correlates. For this purpose, the calibration
will be related with the direction of the incident waves and their shape (local wind
generated waves or ground swells).

# Statement of need 

`CalValWaves` is a Python package for wave reanalysis calibration. Python is a very
high level language that allows the code to be easily understood and changed if
required. All the operations have been performed using commonly known python libraries
such us scipy, numpy and pandas. These friendly features make the usage of the
package available to the all oceanography researchers that are not familiar with the
python language, although use all the potential of it. 

`CalValWaves` was designed to be used by both oceanography researchers and by
students in courses on oceanography and related studies. It was used in a master's
thesis, which results can be seen in the main repository, as the package allows the
creation of very useful plots for the data analtsis. The combination of speed,
and design in `CalValWaves` enables exciting scientific explorations of wave 
reanalysis data all over the world by students and experts alike.

# Mathematics

A simple linear regression is performed that takes into account the existent
different types of waves that appear at the same moment in the wave reanalysis, 
using all of them, which do a total of 32 coefficients (16 directions · 2 types of
waves), to estimate the bulk significant wave height of the waves, measured by the
satellite radars. This linear regression equation can be summarized as follows:

$$
H^{Sat2} = \sum_{i=0}^{15}a_i^2 H_i^{Sea2} + \sum_{i=0}^{15}b_i^2 H_i^{Swell2} + \epsilon^2
$$

where the linear regression is applied over the squares of the significant wave heights,
as the energy is what can be compared in summations, but not just the height. With this
simple linear regression, 32 coefficients are obtained that give an idea about what
waves are over and underestimate by the numerical wave reanalysis.

# Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

# Double dollars make self-standing equations:

# $$\Theta(x) = \left\{\begin{array}{l}
# 0\textrm{ if } x < 0\cr
# 1\textrm{ else}
# \end{array}\right.$$

# You can also use plain \LaTeX for equations
# \begin{equation}\label{eq:fourier}
# \hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
# \end{equation}
# and refer to \autoref{eq:fourier} from text.

# Figures

Some figures can be obtained using this tool, but the most important ones are the
calibration of the hindcast and the validation with the buoy, all shown below in


![calibration.\label{fig:calibration}](/images/calibration-satellite.png)
![validation.\label{fig:validation}](/images/validation-satellite.png)
![comparison.label{fig:comparison}](/images/comparison-satcorr-2008.png)

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
