Introduction
============

The gpvisc package is a companion code for the following publication:

Le Losq C., Ferraina C., Boukaré C.E., Sossi P. (2024) A generalist machine learning model of aluminosilicate melt viscosity: application for exploring the surface properties of the *55 Cancri e* magma ocean. ArXiV TO ADD

In this paper, we describe a new database of phospho-alumino-silicate melt viscosity. Using it, we train and test several machine learning algorithms. We demonstrate that combining a greybox artificial neural network with a Gaussian process provides good results. We apply the new model and various phase diagram calculations to explore the surface properties of the exoplanet `K2-141 b <https://science.nasa.gov/exoplanet-catalog/k2-141-b/>`_.

The code to replicate the analysis performed in the paper is available in the folder `code_paper_EPSL` of the Github repository.

The database
------------

The handheld database assembled from a manual survey of the existing litterature is available at XXXXX. It contains 16,667 published experimental viscosity measurements for melts in the system SiO\ :sub:`2`\-TiO\ :sub:`2`\-Al\ :sub:`2`\ O\ :sub:`3`\-FeO-Fe\ :sub:`2`\O\ :sub:`3`\-MnO-Na\ :sub:`2`\O-K\ :sub:`2`\O-MgO-CaO-P\ :sub:`2`\O\ :sub:`5`\-H\ :sub:`2`\O, spanning superliquidus to undercooled temperatures and pressures up to 30 GPa. When available, the fractions of Fe as FeO and Fe\ :sub:`2`\O\ :sub:`3`\ were compiled. When not available, they were calculated using the Borisov model; in the case no oxygen fugacity details were provided in the publications, we assumed that melts viscosity were measured in air. We provide a function to easily calculate the ratio of ferrous and ferric iron using this model, see the documentation about this feature here: :doc:`inputs`.

We also added experimental viscosity data from SciGlass, for melt compositions that were not appearing already in the handheld database. For that, we use the `GlassPy <https://github.com/drcassar/glasspy/tree/master>`_ library.  A total of 12,231 data points from 3,591 different melt compositions were added. 

The final database contains 28,868 viscosity measurements. It includes data from unary, binary, ternary, quaternary, and multivalent melt compositions, at temperatures spanning superliquidus to undercooled conditions and pressures up to 30 GPa. 

The data are available in the folder `code_paper_EPSL/additional_data <https://github.com/charlesll/gpvisc/tree/master/code_paper_EPSL/additional_data>`_ and here:

Ferraina, Clement; Baldoni, Barbara; Le Losq, Charles, 2024, "Silicate melt viscosity database for gpvisc", `https://doi.org/10.18715/IPGP.2024.lycv4gsa <https://doi.org/10.18715/IPGP.2024.lycv4gsa>`_, IPGP Research Collection, DRAFT VERSION 

The GP model
------------

The GP model combines a greybox artificial neural network (ANN) with a Gaussian process.

Gaussian processes (GPs) are collections of random variables, each being described by a Gaussian distribution. It is frequently said that they allow placing a probability distribution over functions. The mean and covariance (a.k.a. kernel) functions of a GP fully describe it. There is extensive litterature online about GPs, to which we refer the user. As we use GPyTorch, please have a look at their `documentation <https://gpytorch.ai/>`_.

The mean function of a GP usually is choosen as a constant. However, other possibilities exist, as indicated in Rasmussen (2006) or more recent publications. For our work, we chose to use a greybox artificial neural network as the mean function of the GP. The greybox artificial neural network embeds the Vogel-Tamman-Fulcher equation, allowing us to place a prior idea on the functional form of the viscosity versus temperature relationship. 

The GP then can be seen as a method correcting the errors made by the greybox artificial neural network (Rasmussen, 2006).

The model is implemented using `GPyTorch <https://gpytorch.ai/>`_ and `PyTorch <https://pytorch.org/>`_.

With gpvisc, predictions can be made using the GP model, but also directly using the greybox ANN model as it also provides sensible results.