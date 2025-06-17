import streamlit as st
import pandas as pd
import gpvisc
import plotly.graph_objects as go
from scipy.optimize import curve_fit

# Set page title
st.set_page_config(page_title="gpvisc", layout="wide")

# Title
st.title('gpvisc: Melt Viscosity Calculator')

st.markdown("""
            (c) Le Losq C. and co. 2024-2025
            
            **gpvisc is a Python library providing greybox neural network and Gaussian process models
            for the prediction of the viscosity of water-bearing phospho-alumino-silicate melts.**

            **This is an easy to use GUI interface.** Warning: it can be slow to load due to the speed of Streamlit servers.

            Change the parameters on the left panel. For the model, you can select between:

            - the Gaussian Process model - a bit slower but excellent accuracy, and provides error bars.
                
            - the Artificial Neural Network model - faster (x10) but slightly less accurate in average, and do not provide error bars.
            
            You can also query the outputs of three models to check for extrapolation : if they agree within error bars, predictions are robust.

            **A python package is also available. For more information, see**
            - [the gpvisc package documentation](https://gpvisc.readthedocs.io).
            - [the example notebooks](https://github.com/charlesll/gpvisc/tree/master/examples).
            - [check the paper on EPSL](https://doi.org/10.1016/j.epsl.2025.119287),
            - [have a look at the Github repo](https://github.com/charlesll/gpvisc)            
            """)

# Add information about the app
st.sidebar.info('Enter below the melt composition, temperature, and pressure. Then, indicate if you want to query predictions from only model 1 (the best), or if you want to check for extrapolation (see paper) by comparing results from two or three models (slower!).')

st.sidebar.markdown('---')

# Sidebar for composition input type selection
composition_type = st.sidebar.radio("Select composition input type:", ("wt%", "mol%"))

# Function to normalize composition
def normalize_composition(composition_dict):
    total = sum(composition_dict.values())
    return {oxide: value / total * 100 for oxide, value in composition_dict.items()}

# Sidebar for inputs
st.sidebar.header(f'Composition Input ({composition_type})')
composition = {
    'SiO2': st.sidebar.number_input('SiO2', value=60.0, min_value=0.0, max_value=100.0),
    'TiO2': st.sidebar.number_input('TiO2', value=0.0, min_value=0.0, max_value=100.0),
    'Al2O3': st.sidebar.number_input('Al2O3', value=9.0, min_value=0.0, max_value=100.0),
    'FeO': st.sidebar.number_input('FeO', value=10.0, min_value=0.0, max_value=100.0),
    'Fe2O3': st.sidebar.number_input('Fe2O3', value=0.0, min_value=0.0, max_value=100.0),
    'MnO': st.sidebar.number_input('MnO', value=0.0, min_value=0.0, max_value=100.0),
    'Na2O': st.sidebar.number_input('Na2O', value=5.0, min_value=0.0, max_value=100.0),
    'K2O': st.sidebar.number_input('K2O', value=5.0, min_value=0.0, max_value=100.0),
    'MgO': st.sidebar.number_input('MgO', value=10.0, min_value=0.0, max_value=100.0),
    'CaO': st.sidebar.number_input('CaO', value=0.0, min_value=0.0, max_value=100.0),
    'P2O5': st.sidebar.number_input('P2O5', value=0.0, min_value=0.0, max_value=100.0),
    'H2O': st.sidebar.number_input('H2O', value=0.0, min_value=0.0, max_value=100.0)
}

# Temperature and Pressure inputs
st.sidebar.header('Temperature and Pressure')
T_init = st.sidebar.number_input('Initial Temperature (K)', value=1050.0)
T_final = st.sidebar.number_input('Final Temperature (K)', value=2000.0)
P_init = st.sidebar.number_input('Initial Pressure (GPa)', value=0.0)
P_final = st.sidebar.number_input('Final Pressure (GPa)', value=0.0)

# Other parameters
control_redox = st.sidebar.checkbox('Control Redox', value=True)
fo2_init = st.sidebar.number_input('Initial fO2', value=-7.0)
fo2_final = st.sidebar.number_input('Final fO2', value=-1.0)
nb_values = st.sidebar.number_input('Number of data points', value=50, min_value=2, max_value=1000, step=1)

# Model selection
model_type = st.sidebar.radio("Select model type:", ("Gaussian Process", "Artificial Neural Network"))

models_to_use = st.sidebar.multiselect(
    'Select models to use (test for extrapolation):',
    ['Model 1', 'Model 2', 'Model 3'],
    default=['Model 1']
)

# Main calculation functions
@st.cache_data
def prepare_input_data(normalized_composition, composition_type, T_init, T_final, P_init, P_final, control_redox, fo2_init, fo2_final, nb_values):
    Inputs_ = gpvisc.generate_query_single(
        sio2=normalized_composition['SiO2'], 
        tio2=normalized_composition['TiO2'],
        al2o3=normalized_composition['Al2O3'], 
        feo=normalized_composition['FeO'],
        fe2o3=normalized_composition['Fe2O3'], 
        mno=normalized_composition['MnO'],
        na2o=normalized_composition['Na2O'], 
        k2o=normalized_composition['K2O'],
        mgo=normalized_composition['MgO'], 
        cao=normalized_composition['CaO'],
        p2o5=normalized_composition['P2O5'], 
        h2o=normalized_composition['H2O'],
        composition_mole=(composition_type == "mol%"),
        T_init=T_init, T_final=T_final,
        P_init=P_init, P_final=P_final, control_redox=control_redox,
        fo2_init=fo2_init, fo2_final=fo2_final, nb_values=nb_values
    )

    # Scaling
    tpxi_scaled = gpvisc.scale_for_gaussianprocess(
        Inputs_.loc[:,"T"],
        Inputs_.loc[:,"P"],
        Inputs_.loc[:,gpvisc.list_oxides()]
    )

    return Inputs_, tpxi_scaled

# Load models
@st.cache_resource  # <-- Better choice for ML models
class load_viscosity_model():
    """load all models"""

    def __init__(self):
         # Loading the model
         # CPU or GPU?
        self.device = gpvisc.get_device()
        self.model_list = {"1": gpvisc.load_gp_model(model_number=1),
                           "2": gpvisc.load_gp_model(model_number=2),
                           "3":gpvisc.load_gp_model(model_number=3)}

# Load once at app startup
viscosity_models = load_viscosity_model()

# Calculate button
if st.button('Calculate Viscosity'):
    # Normalize the composition
    normalized_composition = normalize_composition(composition)
    
    # Display normalized composition
    st.subheader('Normalized Composition')
    st.write(pd.DataFrame([normalized_composition]).T.rename(columns={0: f'Normalized {composition_type}'}))
    
    # Prepare input data (this is now done only once)
    Inputs_, tpxi_scaled = prepare_input_data(
    normalized_composition,
    composition_type,
    T_init, T_final,
    P_init, P_final,
    control_redox,
    fo2_init, fo2_final,
    nb_values
    )

    # Calculate viscosity for selected models
    results = {}
    for model in models_to_use:
        model_number = model.split()[-1]
        if model_type == "Gaussian Process":
            gp, likelihood = viscosity_models.model_list[model_number]
            results[model] = gpvisc.predict(tpxi_scaled, gp, likelihood)
        else:
            gp, likelihood = viscosity_models.model_list[model_number]
            results[model] = gpvisc.predict(tpxi_scaled, gp, likelihood, model_to_use="ann")

    # Create Plotly figure
    fig = go.Figure()

    # Color map for different models
    colors = {'Model 1': 'red', 'Model 2': 'blue', 'Model 3': 'green'}

    # Add traces for selected models
    if model_type == "Gaussian Process":
        for model, (visco_mean, visco_std) in results.items():
            fig.add_trace(go.Scatter(x=Inputs_.loc[:,"T"], y=visco_mean,
                            mode='lines', name=f'{model} Mean',
                            line=dict(color=colors[model])))
            fig.add_trace(go.Scatter(x=Inputs_.loc[:,"T"], y=visco_mean-visco_std,
                            mode='lines', name=f'{model} Lower Bound (1-sigma)',
                            line=dict(color=colors[model], dash='dash')))
            fig.add_trace(go.Scatter(x=Inputs_.loc[:,"T"], y=visco_mean+visco_std,
                            mode='lines', name=f'{model} Upper Bound (1-sigma)',
                            line=dict(color=colors[model], dash='dash')))
    else:
        for model, visco_mean in results.items():
            fig.add_trace(go.Scatter(x=Inputs_.loc[:,"T"], y=visco_mean,
                            mode='lines', name=f'{model} Mean',
                            line=dict(color=colors[model])))

    # Create buttons for model visibility toggle
    buttons = []
    for i, model in enumerate(models_to_use):
        visibility = [False] * len(fig.data)
        visibility[i*3:(i+1)*3] = [True, True, True]  # Make visible the 3 traces for this model
        buttons.append(dict(label=model,
                            method='update',
                            args=[{'visible': visibility},
                                  {'title': f'Melt Viscosity vs Temperature - {model}'}]))
    
    # Add "Show All" button
    buttons.append(dict(label='Show All',
                        method='update',
                        args=[{'visible': [True] * len(fig.data)},
                              {'title': 'Melt Viscosity vs Temperature - All Models'}]))

    # Update layout to include buttons
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            active=-1,
            x=0.57,
            y=1.2,
            buttons=buttons,
        )]
    )

    fig.update_layout(
        title='Melt Viscosity vs Temperature',
        xaxis_title='Temperature (K)',
        yaxis_title='Viscosity (log₁₀ Pa·s)',
        legend_title='Legend'
    )

    # Display the plot
    st.plotly_chart(fig)

    # Display data for selected models

    # we will perform a quick VFT fit of the tabular data
    # we assume an infinite viscosity of -4.71
    VFT_Acte = lambda T, A, B, C : gpvisc.VFT(T, A, B, C)

    if model_type == "Gaussian Process":
        for model, (visco_mean, visco_std) in results.items():
            st.subheader(f'Calculated Data for {model}')
            df_result = pd.DataFrame({
                'Temperature (K)': Inputs_.loc[:,"T"],
                'Viscosity (log₁₀ Pa·s)': visco_mean,
                'Standard Deviation': visco_std
            })
            st.dataframe(df_result)

            # # VFT calc
            # st.write("VFT parameters are")
            # popt, pcov = curve_fit(gpvisc.VFT, Inputs_.loc[:,"T"], visco_mean)
            # st.write('A : {:.2f}, B: {:.1f}, C: {:.1f}'.format(popt[0], popt[1], popt[2]))
            
            # st.write("VFT fitting error")
            # from sklearn.metrics import root_mean_squared_error as rmse
            # st.write(rmse(VFT_Acte(Inputs_.loc[:,"T"], popt[0], popt[1], popt[2]), visco_mean))
    else:
        for model, visco_mean in results.items():
            st.subheader(f'Calculated Data for {model}')
            df_result = pd.DataFrame({
                'Temperature (K)': Inputs_.loc[:,"T"],
                'Viscosity (log₁₀ Pa·s)': visco_mean
            })
            st.dataframe(df_result)

            # # VFT calc
            # popt, pcov = curve_fit(gpvisc.VFT, Inputs_.loc[:,"T"], visco_mean, p0=[-4.71, 8000, 500])
            # st.warning("Parameters of the VFT equation A + B/(T-C) are A : {:.2f}, B: {:.1f}, C: {:.1f}. Those are adjusted for interpolation of the tabular values only! In general, prefer using directly the outputs of the model 1.".format(popt[0],popt[1],popt[2]))

            # st.write("VFT fitting error")
            # from sklearn.metrics import root_mean_squared_error as rmse
            # st.write(rmse(VFT_Acte(Inputs_.loc[:,"T"], popt[0], popt[1], popt[2]), visco_mean))

    
    
    
    
    