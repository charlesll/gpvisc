.. GP-melt documentation master file, created by
   sphinx-quickstart on Sat May 25 08:14:49 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to gpvisc's documentation!
===================================

Copyright (2024-2025) C. Le Losq and co.

gpvisc is a machine learning model trained to predict the viscosity of phospho-alumino-silicate melts.

It is trained on a extensive database, comprising more than 5,000 different melt compositions for a total of more than 28,000 viscosity data points. For some compositions like peridotite, predictions are even possible at pressures up to 30 GPa. The database is available here. The code is open source on `Github.com/charlesll/gpvisc <https://github.com/charlesll/gpvisc>`_.

To use the model, we provide gpvisc as a Python library. Jupyter Notebooks are provided as examples of use. A easy-to-use no-code web calculator is also available on Streamlit: `https://gpvisc.streamlit.app/ <https://gpvisc.streamlit.app/>`_

Please follow the tutorials of this documentation to perform predictions!

If you use this package, please cite it using those citation keys:

Le Losq, C., Ferraina, C., Sossi, P.A., and Boukaré, C.-É. (2025) A general machine learning model of aluminosilicate melt viscosity and its application to the surface properties of dry lava planets. Earth and Planetary Science Letters, 656, 119287. `https://doi.org/10.1016/j.epsl.2025.119287 <https://doi.org/10.1016/j.epsl.2025.119287>`_

Le Losq C., Ferraina C., Sossi P. A., Boukaré C.-É. (2024). gpvisc. Zenodo. `https://doi.org/10.5281/zenodo.13843250 <https://doi.org/10.5281/zenodo.13843250>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   inputs
   predictions
   tutorials 
   web
   authors
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
