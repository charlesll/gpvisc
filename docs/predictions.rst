Predictions
===========

Making a prediction
-------------------

After data preparation (see :ref:`Inputs`), you first need to import the model:

.. code-block:: python 

    gp_model, likelihood = gpvisc.load_gp_model(model_number=1, device="cpu")

You can use a GPU if you want too (predictions are around 10 times faster on GPU than on CPU). To do so, you can use the helper function to get the GPU

.. code-block:: python

    device = gpvisc.get_device()

at the beginning of your code. Make sure your GPU is available and detected by PyTorch prior to this. Beware that if you GPU has a small memory (4 Go or less), you may run into memory overflow problems. In this case, use the CPU.

In general, we encourage the use of a GPU as we saw differences in the predictive performance when querying results from calculations performed on the CPU and on the GPU. We assign those to floating point errors that slightly affect the performance of the models. Best performance are associated with the use of the GPU.

You can also import a different model number (1, 2 or 3), if you want to test them. In interpolation, they offer very similar results but upon extrapolation, the results diverge significantly. You can try this to see if results are consistent and thus reliable, or not (extrapolation case).

After importing the model, make your predictions using the `gpvisc.predict` function. It takes in input a numpy array with your temperature, pressure and composition. It handles automatically tranfers to torch.FloatTensor type and to the GPU, if you indicate `device="GPU"` for instance in the function (this is an optinal argument).

.. code-block:: python

    viscosity_mean, viscosity_std = predict(tpxi_scaled, gp_model, likelihood, device = device)

Note that you can also query predictions from the mean Greybox artificial neural network that is used in the GP model as the mean function. This model provides predictions that are a little bit less accurate, but inference time are much faster:

.. code-block:: python

    viscosity = predict(tpxi_scaled, gp_model, likelihood, model_to_use = "ann", device = device)

In this case, only one value per query is returned. No uncertainty determination is performed.

Prediction outputs
------------------ 

The `gpvisc.predict` function outputs the mean and the standard deviation of the Gaussian process. Therefore, in the above example, `viscosity_mean` is the predicted viscosity in $\log_{10}$ Pa$\cdot$s and `viscosity_std` is the associated standard error.

For further details, please consult the Simple_prediction.ipynb and the Peridotite.ipynb Jupyter notebooks, see the section :doc:`tutorials`.

Checking for extrapolation
--------------------------

ML algorithms are not very good at performing extrapolation, i.e. at providing estimates for inputs that are outside the range of the training data. At room pressure, the database of the `gpvisc`` model is very broad in terms of compositions and temperature. It thus is unlikely that users ask for queries that are in the "extrapolative regime", i.e. querying viscosity predictions for compositions and temperatures outside the domain covered by the database. But it can happen. At high pressure, this is even more true because the high pressure portion of the database is sparse, owing to the rarity of high pressure viscosity data on silicate melts.

Therefore, when asking for predictions for "exotic" compositions or at high pressure, we recommend checking that the model is not extrapolating. To do so, one simply compares the predictions of the three provided GP models. If predictions are comparable within error bars, they are robust. If they diverge significantly, this indicates that the model is extrapolating and that results are probably not robust. 

For code implementation, please see the notebook `Extrapolations.ipynb <https://github.com/charlesll>`_.

