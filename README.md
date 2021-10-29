# CoCoPy

This repository contains all Jupyter notebooks and scripts needed to reproduce or extend the analysis presented in the following study.

R. Andrassy, J. Higl, H. Mao, M. Mocák, D. G. Vlaykov, W. D. Arnett, I.
Baraffe, S. W. Campbell, T. Constantino, P. V. F. Edelmann, T. Goffrey , T.
Guillet, F. Herwig, R. Hirschi, L. Horst, G. Leidi, C. Meakin, J. Pratt, F.
Rizzuti, F. K. Röpke, and P. Woodward 2021, **Dynamics in a stellar convective layer and at its boundary: Comparison of five 3D hydrodynamics codes**, submitted to Astronomy & Astrophysics in October 2021

The Jupyter notebooks can be run on the [CoCo Hub](https://www.ppmstar.org/coco/), which also contains the data files necessary for the analysis. Alternatively, you can clone this repository on your machine and download the 1D and 2D data from [Zenodo](https://doi.org/10.5281/zenodo.5607675). The original 3D data cubes are only available on the [CoCo Hub](https://www.ppmstar.org/coco/) because of the large data volume.

## What is included

* `analysis.py`: Python script that was used to generate 1D averages, spectra and 2D slices from the original 3D data cubes.
* `1D2D/`
    * `1D-profiles.ipynb`: Jupyter notebook that analyses and plots the 1D profiles. The notebook also plots the MESA stellar-structure model contained in the file `mesa_model_29350.npy`, which the simulation setup is based on.
    * `2D-slices.ipynb`: Jupyter notebook that compares 2D slices from different simulations at a given point in time.
    * `spectra.ipynb`: Jupyter notebook that loads and plots kinetic-energy spectra.
    * `animations.ipynb`: Jupyter notebook that creates individual frames of the animations available on [Zenodo](https://doi.org/10.5281/zenodo.5607675).
    * `FK-in-two-stream-model.ipynb`: Jupyter notebook that creates Fig. B.1 shown in the paper's Appendix B.
    * `setup-two-layers.ipynb`: Jupyter notebook that generates the 1D initial stratification as defined in the paper.
    * `plotting_settings.py`: Python script that that defines the most important plotting-related parameters and variables.
    * `make-hstry-files.ipynb`: Jupyter notebook that generates simulation-history files. Although not necessary for the analysis, the history files speed up access to the `.rprof` files by simulation time, which may be useful in future studies.
* `3D/`
    * `3D-data-exploration.ipynb`: Jupyter notebook that shows how to load and visualise the 3D data.
    * `PPMstardata.py`: Python reader of 3D data cubes from the PPMSTAR code.
    * `slhoutput.py`: Python reader of 3D data cubes from the SLH code.
    * `UTILS/PROMPI/PROMPI_data.py`: Python reader of 3D data cubes from the PROMPI code.