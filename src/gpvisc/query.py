import gpvisc
import numpy as np
import pandas as pd

######################################
## HELPER FUNCTIONS FOR PREDICTIONS ##
######################################

def generate_query_single(sio2=100.0,
                          tio2=0.0,
                          al2o3=0.0,
                          feo=0.0,
                          fe2o3=0.0,
                          mno=0.0,
                          na2o=0.0,
                          k2o=0.0,
                          mgo=0.0,
                          cao=0.0,
                          p2o5=0.0,
                          h2o=0.0,
                          composition_mole = True,
                          T_init = 1400, T_final= 1400,
                          P_init = 0.0, P_final = 0.0,
                          control_redox = False,
                          fo2_init = -12.0, fo2_final = -5.0,
                          redox_model="B2018",
                          nb_values = 100, 
                          ):
    """Generates a query DataFrame for a single magma composition 
       with multiple P/T values.

    Parameters
    ----------
        sio2, tio2, ..., h2o: Oxide weight percentages (if composition_mole=False) or 
            mole fractions (if composition_mole=True). Defaults to pure SiO2.
        composition_mole: Boolean indicating if input composition is in mole fraction (True) 
            or weight percent (False). Defaults to True.
        T_init, T_final: Initial and final temperatures (in Kelvin) for the query. Defaults to 1400 K.
        P_init, P_final: Initial and final pressures (in GPa) for the query. Defaults to 0 GPa.
        control_redox: Boolean indicating if the Fe redox state should be controlled. Defaults to False.
        fo2_init, fo2_final: Initial and final log10(fO2) values for redox control (only if control_redox=True).
        redox_model : string indicating which model to use. See the documentation of the function redox() for details. Default: "B2018".
        nb_values: Number of P/T points to generate within the specified ranges. Defaults to 100.

    Returns
    -------
        pd.DataFrame: A DataFrame containing the query with columns for T, P, and oxide compositions.
    """
    
    db = pd.DataFrame({
                      "T": np.linspace(T_init, T_final, nb_values),
                      "P": np.linspace(P_init, P_final, nb_values)
                      })
    
    for oxide in gpvisc.list_oxides():
        db[oxide] = locals()[oxide]  # Elegant way to set values from function arguments

    if db.loc[0, gpvisc.list_oxides()].sum() != 100.0:
        print("Warning: Composition does not sum to 100%. Renormalizing...")

    # convert in fractions and normalise.
    db = gpvisc.chimie_control(db).copy()

    if not composition_mole:
        print("Converting weight percent composition to mole fraction...")
        db = gpvisc.wt_mol(db) 

    # control the oxidation state of Fe
    if control_redox == True: 
        print("Calculation iron redox...")
        db = generate_query_redox(db, 
                                  np.linspace(fo2_init, fo2_final, nb_values), 
                                  model=redox_model)
    
    return db

def generate_query_redox(db, fo2_vector, model="B2018"):
    """Recalculates FeO and Fe2O3 mole fractions in a DataFrame based on oxygen fugacity (fO2).

    This function applies the Borisov 2018 model to determine the Fe3+/ΣFe ratio 
    for each row in the input DataFrame `db`, given the corresponding fO2 values in 
    `fo2_vector`. The FeO and Fe2O3 mole fractions are then adjusted accordingly, 
    maintaining the total iron content. Finally, the composition is renormalized.

    Parameters
    ----------
        db : pandas DataFrame containing melt composition data, including columns for 'feo', 'fe2o3', and 'T' (temperature).
        fo2_vector :  NumPy array or list containing log10(fO2) values corresponding to each row in `db`.
        model : string indicating which model to use. See the documentation of the function redox() for details. Default: "B2018".
    Returns:
        pandas DataFrame: A copy of the input DataFrame with updated FeO and Fe2O3 mole fractions
                          and renormalized composition.

    Raises
    ------
        ValueError: If the lengths of `db` and `fo2_vector` do not match.

    Note
    ----
        This function assumes that the input DataFrame `db` is already in mole fraction units.
    """
    
    # Input Validation:
    if len(db) != len(fo2_vector):
        raise ValueError("Length of fo2_vector must match the number of rows in db.")

    db2 = db.copy()

    # Calculate Redox Ratio (Fe3+/ΣFe) using the redox model
    #redox_ = gpvisc.redox_B2018(db2, 10**fo2_vector, db["T"])
    redox_ = gpvisc.redox(db2, 10**fo2_vector, db["T"], model=model, P=db["P"])
    
    # Calculate total FeO (all Fe as FeO)
    total_iron_feo = db2["feo"] + 2 * db2["fe2o3"] 

    # Recalculate Fe2O3 and FeO mole fractions based on redox_
    db2["fe2o3"] = redox_ * total_iron_feo / 2.0  
    db2["feo"] = (1.0 - redox_) * total_iron_feo

    return gpvisc.chimie_control(db2).copy()  # Renormalize and return a copy

def generate_query_range(
    oxide_ranges: dict,  # Use a dictionary to specify oxide ranges
    composition_mole=True,
    T_init=1400, T_final=1400, P_init=0.0, P_final=0.0,
    control_redox=False, fo2_init=-12.0, fo2_final=-5.0, 
    redox_model="B2018", nb_values=10,
):
    """Generates a query DataFrame for multiple magma compositions within specified ranges,
       each with multiple P/T values.

    Parameters
    ----------
        oxide_ranges: A dictionary specifying the initial and final values for each oxide.
            Keys should be oxide names (e.g., 'SiO2', 'TiO2'), and values should be lists or tuples 
            of length 2: [min_value, max_value].
        composition_mole: Boolean indicating if input composition is in mole fraction (True) 
            or weight percent (False). Defaults to True.
        T_init, T_final: Initial and final temperatures (in Kelvin) for the query.
        P_init, P_final: Initial and final pressures (in GPa) for the query.
        control_redox: Boolean indicating if the Fe redox state should be controlled.
        fo2_init, fo2_final: Initial and final log10(fO2) values for redox control (only if control_redox=True).
        redox_model : string indicating which model to use. See the documentation of the function redox() for details. Default: "B2018".
        nb_values: Number of values to generate within each oxide range (and for T and P).

    Returns
    -------
        pd.DataFrame: A DataFrame containing the query with columns for T, P, 
                      oxide compositions, and a composition ID.
    """

    if nb_values <= 1:
        raise ValueError("nb_values must be greater than 1 to generate a range of values.")

    # Check if oxide_ranges dictionary contains the required oxides
    for oxide in gpvisc.list_oxides():
        if oxide not in oxide_ranges:
            raise ValueError(f"Missing oxide range: {oxide}")

    # Initialize empty lists for T, P, and compositions
    T_values = np.linspace(T_init, T_final, nb_values)
    P_values = np.linspace(P_init, P_final, nb_values)

    # Create all combinations of oxide values
    oxide_values = [np.linspace(oxide_ranges[oxide][0], oxide_ranges[oxide][1], nb_values) 
                    for oxide in gpvisc.list_oxides()]
    
    db = pd.DataFrame(np.array(oxide_values).T, columns=gpvisc.list_oxides())
    db["T"] = T_values
    db["P"] = P_values

    if db.loc[0, gpvisc.list_oxides()].sum() != 100.0:
        print("Warning: Composition does not sum to 100%. Renormalizing...")

    # convert in fractions and normalise.
    db = gpvisc.chimie_control(db).copy()

    if not composition_mole:
        print("Converting weight percent composition to mole fraction...")
        db = gpvisc.wt_mol(db) 

    # control the oxidation state of Fe
    if control_redox == True: 
        print("Calculation iron redox...")
        db = generate_query_redox(db, 
                                  np.linspace(fo2_init, fo2_final, nb_values), 
                                  model=redox_model)
    
    return db