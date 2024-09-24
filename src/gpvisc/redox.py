# -*- coding: utf-8 -*-
"""
Friday 16 March 2018

@author: Charles LE LOSQ

Calculation of the structure of silicate melts

"""
import numpy as np

def redox(chimie,fo2,T, model="KC1991", P=0.0001):
    """return the ration of Fe3+ over total iron for a melt at given T and fO2, 1 atm pressure

    Parameters
    ==========
    chimie : pandas dataframe
        the melt chemical composition in mol fraction

    fo2 : ndarray
        the oxygen fugacity, size n

    T : ndarray
        the temperature in K, size n

    Options
    -------
    model : string
        The model that should be used. Choose between KC1991, B2018, SY2024.

    P : float or ndarray of size n
        The pressure in GPa. Only used by the SY2024 model.

    Returns
    =======
    fe3_fe3pfe2 : ndarray
        the ratio of Fe3+ over (Fe3+ + Fe2+) in the melt, size n

    References
    ==========
    KC1991 : Kress, V.C., Carmichael, I.S., 1991. The compressibility of silicate liquids 
             containing Fe2O3 and the effect of composition, temperature, oxygen fugacity 
             and pressure on their redox states. 
             Contributions to Mineralogy and Petrology 108, 82–92.
    B2018 : Borisov, A., Behrens, H., Holtz, F., 2018. 
            Ferric/ferrous ratio in silicate melts: a new model for 
            1 atm data with special emphasis on the effects of melt composition. 
            Contrib Mineral Petrol 173, 98. 
            https://doi.org/10.1007/s00410-018-1524-8v
    SY2024 : Sun, C., Yao, L., 2024. 
             Redox equilibria of iron in low- to high-silica melts: 
             A simple model and its applications to C-H-O-S degassing. 
             Earth and Planetary Science Letters 638, 118742. 
             https://doi.org/10.1016/j.epsl.2024.118742
    """

    if model == "KC1991":
        return redox_KC1991(chimie, fo2, T)
    elif model == "B2018":
        return redox_B2018(chimie, fo2, T)
    elif model == "SY2024":
        return redox_SY2024(chimie, fo2, T, P=P)
    else:
        ValueError("Choose between KC1991 and B2018 models")

def redox_B2018(chimie, fo2, T):
    """return the ration of Fe3+ over total iron for a melt at given T and fO2, 1 atm pressure

    It uses the Borisov et al. (2018) parametric model

    Parameters
    ==========
    chimie : pandas dataframe
        the melt chemical composition in mol%

    fo2 : float or ndarray
        the oxygen fugacity, float or array of size n

    T : ndarray
        the temperature in K, size n

    Options
    -------
    model : string
        The model that should be used. Choose between KC1991, B2018

    Returns
    =======
    fe3_fe3pfe2 : ndarray
        the ratio of Fe3+ over (Fe3+ + Fe2+) in the melt, size n

    To Do
    =====
    Raise error if fo2 and T are of different sizes
    """


    a = 0.207
    b = 4633.3
    c = -1.852
    d_sio2 = -0.445
    d_tio2 = -0.900
    d_mgo = 1.532
    d_cao = 0.314
    d_na2o = 2.030
    d_k2o = 3.355
    d_p2o5 = -4.851
    d_sial = -3.081
    d_simg = -4.370

    xdi = (chimie["sio2"]*d_sio2 +
          chimie["tio2"]*d_tio2 +
          chimie["mgo"]*d_mgo +
          chimie["cao"]*d_cao +
          chimie["na2o"]*d_na2o +
          chimie["k2o"]*d_k2o +
          chimie["p2o5"]*d_p2o5 +
          chimie["sio2"]*chimie["al2o3"]*d_sial +
          chimie["sio2"]*chimie["mgo"]*d_simg)

    log_feo1p5_feo = a*np.log10(fo2) + b/T + c + xdi # 1 atm equation

    fe3_fe3pfe2 = 10**log_feo1p5_feo/((1+10**log_feo1p5_feo)) #

    return fe3_fe3pfe2


def redox_KC1991(chimie,fo2, T):
    """return the ration of Fe3+ over total iron for a melt at given T and fO2, 1 atm pressure

    It uses the Kress and Carmnichael (1991) parametric model

    Parameters
    ==========
    chimie : pandas dataframe
        the melt chemical composition in mol%

    fo2 : float or ndarray
        the oxygen fugacity, float or array of size n

    T : ndarray
        the temperature in K, size n

    Returns
    =======
    fe3_fe3pfe2 : ndarray
        the ratio of Fe3+ over (Fe3+ + Fe2+) in the melt, size n

    To Do
    =====
    Raise error if fo2 and T are of different sizes
    """

    a = 0.196
    b = 1.1492*10**4
    c = -6.675
    d_al2o3 = -2.243
    d_feo = -1.828
    d_cao = 3.201
    d_na2o = 5.854
    d_k2o = 6.215
    e = -3.36
    f = -7.01*10**-7
    g = -1.54*10**-10
    h = 3.85*10**-17

    xdi = chimie["al2o3"]*d_al2o3 + chimie["feo"]*d_feo + chimie["cao"]*d_cao + chimie["na2o"]*d_na2o + chimie["k2o"]*d_k2o

    ln_fe2o3_feo = a*np.log(fo2) + b/T + c + xdi # 1 atm equation

    fe3_fe3pfe2 = np.exp(ln_fe2o3_feo+np.log(2))/((1+np.exp(ln_fe2o3_feo+np.log(2)))) # np.log(2)factor two because I want XFeO and XFeO1.5 to have Fe3+/Fe2+

    return fe3_fe3pfe2

def t_calc(P, params, P0 = 0.0001):
    """calculation of coefficient t_i
    P : ndarray
        pressure in GPa 
    params: dict
        coefficients b and c for t_i, see Table S3

    P0 is the reference pressure in GPa"""

    term0 = params["b0"]*P**2*np.log(P/P0)

    term1to4 = np.zeros(len(P))
    for n in range(1,5):
        term1to4 += params["b"+str(n)] * (P - P0)**n + params["c"+str(n)] * (P - P0)**(n - 0.5)

    return term0 + term1to4

def Gamma(P,T):
    """calculation of gamma(P,T)

    Parameters from Table S3
    
    """
    t0_ = {"b0" : -1.75528E-01,
          "b1" : 3.48174E+00,
          "b2" : 3.06370E+00,
          "b3" : 1.36134E-02,
          "b4" : 1.52660E-05,
          "c1" : -4.68802E-01,
          "c2" : -3.58957E+00,
          "c3" : -1.09496E-01,
          "c4" : -7.28938E-04,
          }
    t1_ = {"b0" : 1.82549E-03,
          "b1" : -1.06395E-02,
          "b2" : -2.36645E-02,
          "b3" : -1.56206E-08,
          "b4" : -1.66849E-08,
          "c1" : 1.44394E-03,
          "c2" : 1.48791E-02,
          "c3" : -3.32256E-04,
          "c4" : 5.45464E-07,
          }
    t2_ = {"b0" : -2.14783E-04,
          "b1" : 1.19184E-03,
          "b2" : 2.76222E-03,
          "b3" : -3.92864E-07,
          "b4" : 1.56116E-09,
          "c1" : -1.60439E-04,
          "c2" : -1.69242E-03,
          "c3" : 4.31406E-05,
          "c4" : -4.43921E-08,
          }

    t0 = t_calc(P, t0_)
    t1 = t_calc(P, t1_)
    t2 = t_calc(P, t2_)

    return t0 + t1 * T + t2 * T * np.log(T)

def redox_SY2024(chimie, fo2, T, P = 0.0001):
    """calculates Fe3+/(Fe3+ + Fe2+) ratio in silicate melts using the Sun and Yao (2024) model.

    Parameters
    ==========
    chimie : pandas dataframe
        the melt chemical composition in mol fraction

    fo2 : float or ndarray
        the oxygen fugacity, float or array of size n

    T : ndarray
        the temperature in K, size n

    P : float or ndarray
        the pressure in GPa, float or ndarray of size n. Default = 0.0001 (1 bar).

    Returns
    =======
    fe3_fe3pfe2 : ndarray
        the ratio of Fe3+ over (Fe3+ + Fe2+) in the melt, size n
       
    """

    # Model parameters (from Table 2)
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, h = (
        2.1479, -230.2593, -1.8557e-4, 34.3293, 1.4138, -17.3040, -10.1820,
        -6.7463, -7.3886, -14.5430, -9.9776, -16.1506, -37.5572, 2.1410
    )

    # get the chemical composition in the good shape
    X_SiO2 = chimie["sio2"] 
    X_TiO2  = chimie["tio2"]
    X_AlO1_5  = chimie["al2o3"]/2.0
    X_FeO  = chimie["feo"]
    X_MgO  = chimie["mgo"]
    X_CaO  = chimie["cao"]
    X_NaO0_5  = chimie["na2o"]/2.0
    X_KO0_5 = chimie["k2o"]/2.0

    # renormalization
    X_sum = X_SiO2 + X_TiO2 + X_AlO1_5 + X_FeO + X_MgO + X_CaO + X_NaO0_5 + X_KO0_5
    for i in [X_SiO2, X_TiO2, X_AlO1_5, X_FeO, X_MgO, X_CaO, X_NaO0_5, X_KO0_5]:
        i = i/X_sum

    # if type(P) == float
    # we transform it into an, array of len(T)
    if type(P) == float:
        P = np.ones(len(T))*P

    # good scale of fo2
    logfo2 = np.log10(fo2)

    # now calculate Omega and Phy
    Omega_T = a1 + a2*(T**1.5) + a3*np.log(T)
    
    Phy_X = (a4 * np.log(X_FeO) 
             + a5 * (X_FeO**0.5)
             + a6 * (X_SiO2**3)
             + a7 * X_AlO1_5
             + a8 * X_TiO2
             + a9 * X_CaO
             + a10 * X_MgO
             + (a11 + a12*X_FeO) * (X_NaO0_5 + X_KO0_5)
            )
    
    # Final calculation
    #- h * Gamma(P, T)
    Fe3_Fe2 = 10**((logfo2 - Omega_T - Phy_X ) / (4.0 + a0 * (X_FeO**0.5)))
    fe3_fe3pfe2 = Fe3_Fe2/(1.0+Fe3_Fe2)
    
    return fe3_fe3pfe2
