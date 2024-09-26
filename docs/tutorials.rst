Tutorials
==========================

Simple prediction
-----------------

The notebook `Simple_prediction.ipynb <https://github.com/charlesll/gpvisc/blob/master/examples/Simple_prediction.ipynb>`_ shows how you can perform predictions with the model.

Peridotite
----------

The `Peridotite.ipynb <https://github.com/charlesll/gpvisc/blob/master/examples/Peridotite.ipynb>`_ notebook showcases how for a particular composition, here peridotite, one can perform predictions at room and high pressure.

It further shows how it is possible to query points at specific P-T conditions.

Redox
-----

The `Redox.ipynb <https://github.com/charlesll/gpvisc/blob/master/examples/Redox.ipynb>`_ notebook shows how you can calculate the iron oxidation state given T and fO2 conditions, and make predictions as a function of it for four different geological melt compositions.

Benchmark on SciGlass (full Database)
-------------------------------------

The notebook `Benchmark.ipynb <https://github.com/charlesll/gpvisc/blob/master/examples/Benchmark.ipynb>`_ provides an example of how to query SciGlass using GlassPy,a nd then benchmark gpvisc against GlassNet for instance on the phospho-alumino-silicate dataset.

Check extrapolations
--------------------

The notebook `Extrapolations.ipynb <https://github.com/charlesll/gpvisc/blob/master/examples/Extrapolation.ipynb>`_ shows how you can check if the model start extrapolating dangerously, by actually looking at the outputs of the three different provided models. 

Benchmark speed
--------------------

The notebook `Speed_test.ipynb <https://github.com/charlesll/gpvisc/blob/master/examples/Speed_test.ipynb>`_ allows benchmarking the speed of the GP and Greybox ANN models on your CPU and GPU.