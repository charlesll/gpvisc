.. GP-melt documentation master file, created by
   sphinx-quickstart on Sat May 25 08:14:49 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to gpvisc's documentation!
===================================

Copyright (2024) C. Le Losq and co.

Charles Le Losq, Institut de physique du globe de Paris, University Paris Cité
lelosq@ipgp.fr

Clément Ferraina, Institut de physique du globe de Paris, University Paris Cité
ferraina@ipgp.fr

gpvisc is a machine learning model trained to predict the viscosity of phospho-alumino-silicate melts.

It is trained on a extensive database, comprising more than 5,000 different melt compositions for a total of more than 28,000 viscosity data points. For some compositions like peridotite, predictions are even possible at pressures up to 30 GPa. The database is available here. The code is open source on `Github.com/charlesll/gpvisc <https://github.com/charlesll/gpvisc>`_.

To use the model, we provide gpvisc as a Python library. Jupyter Notebooks are provided as examples of use. A Streamlit version is under construction.

Please follow the tutorials of this documentation to perform predictions!

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   inputs
   predictions
   tutorials 
   web
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
