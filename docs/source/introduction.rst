Introduction
============

The database
------------

The handheld database assembled from a manual survey of the existing litterature is available at XXXXX. It is described in the article XXXXXX. It contains XXXXXXX published data for melts containing the oxides SiO\ :sub:`2`\, TiO\ :sub:`2`\, Al\ :sub:`2`\ O\ :sub:`3`\, FeO, Fe\ :sub:`2`\O\ :sub:`3`\, MnO, Na\ :sub:`2`\O, K\ :sub:`2`\O, MgO, CaO, P\ :sub:`2`\O\ :sub:`5`\, H\ :sub:`2`\O, for a total of XXXXX data points. It includes data from unary, binary, ternary, quaternary, and multivalent melt compositions. It further contained 1,113 viscosity measurements for 201 melt compositions at high pressures, up to 30 GPa.

When available, the fractions of Fe as FeO and Fe$_2$O$_3$ were compiled. When not available, they were calculated using the Borisov model; in the case no oxygen fugacity details were provided in the publications, we assumed that melts viscosity were measured in air. We provide a function to easily calculate the ratio of ferrous and ferric iron using this model, see the documentation about this feature here: :doc:`inputs`.

We added also in the final database data from SciGlass, for melt compositions that were not appearing already in the handheld database. For that, we use the `GlassPy<https://github.com/drcassar/glasspy/tree/master>` library. The final database contains XXXXX datapoints for XXXXX melt compositions.

The GP model
------------

The GP model is described in this paper. It is the combination of a greybox artificial neural network with a Gaussian process.

Gaussian processes (GPs) are collections of random variables, each being described by a Gaussian distribution. It is frequently said that they allow placing a probability distribution over functions.

The mean and covariance (a.k.a. kernel) functions of a GP fully describe it. There is extensive litterature online about GPs, to which we refer the user. As we use GPyTorch, please have a look at their documentation but also here and here.

The mean function of a GP usually is choosen as a constant. However, other possibilities exist, as indicated in Rasmussen (2006) or more recent publications (links).

For our work, we chose to use a greybox artificial neural network as the mean function of the GP. The greybox artificial neural network embeds the Vogel-Tamman-Fulcher equation, allowing us to place a prior idea on the functional form of the viscosity versus temperature relationship. 

The GP then can be seen as a method correcting the errors made by the greybox artificial neural network (Rasmussen, 2006).

The model is implemented using `GPyTorch<https://gpytorch.ai/>` and `PyTorch<https://pytorch.org/>`.

References
----------

Le Losq C., Ferraina C., Boukaré C.E., Sossi P. (2024) A generalist machine learning model of aluminosilicate melt viscosity: application for exploring the surface properties of the *55 Cancri e* magma ocean. ArXiV TO ADD

In this paper, we describe the new database. Using it, we train and test several algorithms. We demonstrate that the GP method provides the best results, and we apply the model to calculate the viscosity at the surface of the exoplanet `55 Cancri e<https://science.nasa.gov/exoplanet-catalog/55-cancri-e/>`.

The code to replicate the analysis performed in the paper is available in the folder `code_paper_EPSL` of the Github repository.