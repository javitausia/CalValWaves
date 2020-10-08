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

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
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

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Fenced code blocks are rendered with syntax highlighting:
```python
for n in range(10):
    yield f(n)
```	

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References