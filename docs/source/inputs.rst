Inputs
======

Input composition for GP
------------------------

Input compositions for the GP model are mole fractions of the oxides components SiO\ :sub:`2`\, TiO\ :sub:`2`\, Al\ :sub:`2`\ O\ :sub:`3`\, FeO, Fe\ :sub:`2`\O\ :sub:`3`\, MnO, Na\ :sub:`2`\O, K\ :sub:`2`\O, MgO, CaO, P\ :sub:`2`\O\ :sub:`5`\, H\ :sub:`2`\O, in this respective order.

Indicate a composition in a Python dictionnary
----------------------------------------------

An easy and direct way to input a melt composition is to provide it as a Python dictionnary in a `Pandas <https://pandas.pydata.org/>`_ DataFrame:

.. code-block:: python

    my_composition = {"sio2":np.array([0.60,]),
                    "tio2":np.array([0.02,]),
                    "al2o3":np.array([0.12,]),
                    "feo":np.array([0.07,]),
                    "fe2o3":np.array([0.00]),
                    "mno":np.array([0.00,]),
                    "na2o":np.array([0.03,]),
                    "k2o":np.array([0.03,]),
                    "mgo":np.array([0.07,]),
                    "cao":np.array([0.05,]),
                    "p2o5":np.array([0.00,]),
                    "h2o":np.array([0.01,])}
    ds_ = pd.DataFrame(my_composition)

You could also write a smaller composition, even in mole percent:

.. code-block:: python

    my_composition = {"sio2":np.array([60.0,]),
                      "al2o3":np.array([10.0,]),
                      "mgo":np.array([30.0,])}
    ds_ = pd.DataFrame(my_composition)

In this case, always call the function `gpvisc.chimie_control()` to add the other oxides and make everything right:

.. code-block:: python

    ds_ = gpvisc.chimie_control(ds_)

If your composition is in weight percent, you can call the function `gpvisc.wt_mol()` to convert it:

.. code-block:: python

    ds_ = gpvisc.wt_mol(ds_)

Import a composition from Excel/Libre Office
--------------------------------------------

You can import a spreadsheet with `Pandas <https://pandas.pydata.org/>`_, and then as it does not contain columns for some elements, we always call the function `gpvisc.chimie_control()` to add them and put everything in fractions. Then, if necessary, a convertion in mole fractions can be performed using the `gpvisc.wt_mol()` function. Here an example to import a spreadsheet named "compositions.xlsx":

.. code-block:: python

    db = pd.read_excel("./additional_data/compositions.xlsx")
    db = gpvisc.chimie_control(db)
    db_mol = gpvisc.wt_mol(db)

Dealing with iron oxidation state
---------------------------------

If you need to calculate the redox state of iron, you can use the function `gpvisc.redox`

.. code-block:: python

    absolute_fo2 = 0.21
    T = 1273.0
    redox_model = "B2018"
    gpv.redox(db_mol, absolute_fo2, T, redox_model)

You will then need to recalculate your colums `feo` and `fe2o3` depending on the results of this calculation.

Final preparation of melt composition
-------------------------------------

A final step prior to inputing the melt composition as a query in the GP model is required: the columns need to be in a particular order, and the input composition array needs to only contain 12 columns containing oxides mole fractions. The above Pandas dataframes can contain much more information than that.

This is easily performed using the following command:

.. code-block:: python

    compo_for_GP = db_mol.loc[:, gpvisc.list_oxides()].copy()

The `gpvisc.list_oxides` function contains a list of the oxide components in the good order, such that by using the following Pandas query we obtain a Pandas dataframe with the right columns. We ask for a copy at the end to avoid any problem.

Temperature and pressure
------------------------

Temperature and pressure should be vectors or lists of the same size.

You can build them using the numpy functions `numpy.arange()` or `numpy.linspace()`, see their relevant documentation.

We provide a small example below. We want temperature from 500 to 3000 K, every 1 K, and pressure will be kept constant at 0 GPa = 1 atm.

.. code-block:: python

    temperature_vector = np.arange(500.0,3000.0,1.0)
    pressure_vector = np.zeros(len(temperature_vector))

Helper functions to create queries
----------------------------------

The above steps are automated in two helper functions:

- `generate_query_single` generates a query for a given composition following a range of temperature, pressure and oxygen fugacity conditions.
- `generate_query_range` generates a query for a range of compositions.

You can directly indicate the composition you want in `generate_query_single`. It also handles weight to mol convertion as well as determination of Fe redox state. Here is an example of input of a melt composition in wt%, asking for 50 values at 0 GPa and T between 1050 and 2000 K, and log fO2 between -12 and -5.

.. code-block:: python

    Inputs_ = gpvisc.generate_query_single(sio2=60.0,
                       tio2=0.0,
                       al2o3=9.0,
                       feo=10.0,
                       fe2o3=0.00,
                       mno=0.00,
                       na2o=5.0,
                       k2o=5.0,
                       mgo=10.0,
                       cao=0.0,
                       p2o5=0.00,
                       h2o=0.00,
                       composition_mole=False,
                       control_redox=True,
                       T_init=1050.,
                       T_final=2000.,
                       P_init=0.0,
                       P_final = 0.0,
                       nb_values=50)

In `generate_query_range`, you need to provide a dictionary that contains the range of compositions you want to cover. Here is an example:

.. code-block:: python

    oxide_ranges = {
    'sio2': [50.5, 77.],
    'tio2': [0.0, 0.5],
    'al2o3': [14.7, 13.],
    "feo":[10.4,1.],
    "fe2o3":[0.,0.],
    "mno":[0.,0.],
    "na2o":[2.8,3.4],
    "k2o":[0.2,5.3],
    "mgo":[7.58,0.0],
    "cao":[11.4,0.4],
    "p2o5":[0.0,0.0],
    "h2o":[0.1,5.],
    }

    Inputs_range = gpvisc.generate_query_range(oxide_ranges, 
                                                composition_mole=False,
                                                T_init=1473, T_final=1073, 
                                                P_init=1.0, P_final=0.0,
                                                control_redox=True, 
                                                fo2_init=-7.0, fo2_final=-1.0, 
                                                nb_values=50)

Final preparation for import in GP model
----------------------------------------

To prepare the final array for predictions, use the function `gpvisc.scale_for_gaussianprocess()`:

.. code-block:: python

    X_for_GP = gpvisc.scale_for_gaussianprocess(temperature_vector, pressure_vector, compo_for_GP)

If you used the above described helper functions, here is how you can scale things:

.. code-block:: python

    tpxi_scaled = gpvisc.scale_for_gaussianprocess( 
                               Inputs_.loc[:,"T"], # temperature input
                               Inputs_.loc[:,"P"], # pressure input
                               Inputs_.loc[:,gpvisc.list_oxides()] # composition input
                               )

You are now ready to perform a query using the GP model!