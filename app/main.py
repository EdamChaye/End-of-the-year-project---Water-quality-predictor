import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Define the potable and non-potable ranges
potable_ranges = {
    'ph': (6.5, 8.5),
    'Hardness': (60, 120),
    'Solids': (0, 500),
    'Chloramines': (0, 4),
    'Sulfate': (0, 250),
    'Conductivity': (50, 500),
    'Organic_carbon': (0, 4),
    'Trihalomethanes': (0, 80),
    'Turbidity': (0, 1)
}

non_potable_ranges = {  
    'ph': (5, 9),
    'Hardness': (0, 500),
    'Solids': (0, 1500),
    'Chloramines': (0, 10),
    'Sulfate': (0, 500),
    'Conductivity': (50, 2000),
    'Organic_carbon': (0, 10),
    'Trihalomethanes': (0, 150),
    'Turbidity': (0, 10)
}

# Fill missing values based on the potable and non-potable ranges
def fill_missing_values(row):
    for feature in potable_ranges.keys():
        if pd.isnull(row[feature]):
            if row['Potability'] == 1:
                row[feature] = np.random.uniform(*potable_ranges[feature])
            else:
                row[feature] = np.random.uniform(*non_potable_ranges[feature])
    return row

# Load and clean the dataset
def get_clean_data():
    data = pd.read_csv('C:/Users/hp/Desktop/Streamlit water quality/data/data.csv')

    data = data.apply(fill_missing_values, axis=1)
    return data

# Add sliders to the sidebar
def add_sidebar():

    st.sidebar.header("Mesure des caractéristiques")
    data = get_clean_data()
    
    slider_labels = [
        ("ph", "ph"),
        ("Hardness", "Hardness"),
        ("Solids", "Solids"),
        ("Chloramines", "Chloramines"),
        ("Sulfate", "Sulfate"),
        ("Conductivity", "Conductivity"),
        ("Organic_carbon", "Organic_carbon"),
        ("Trihalomethanes", "Trihalomethanes"),
        ("Turbidity", "Turbidity"),
    ]


    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value=float((data[key].min() + data[key].max())/2)
        )
            
    return input_dict

# Scale the values for radar chart visualization
def get_scaled_values(input_dict):
    data = get_clean_data()
    scaled_dict = {}
    
    for key, value in input_dict.items():
        max_val = data[key].max()
        min_val = data[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    
    return scaled_dict

# Generate a radar chart
def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    
    categories = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['ph'], input_data['Hardness'], input_data['Solids'],
            input_data['Chloramines'], input_data['Sulfate'], input_data['Conductivity'],
            input_data['Organic_carbon'], input_data['Trihalomethanes'], input_data['Turbidity']
        ],
        theta=categories,
        fill='toself',
        name='Valeurs entrées'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )
    
    return fig

# Add predictions based on the input data
def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    
    # Convert input_dict to DataFrame to retain feature names
    input_df = pd.DataFrame([input_data])
    
    # Ensure the input data has the same columns as the scaler was fitted with
    input_array_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_array_scaled)
    
    st.subheader("Prédiction de la potabilité de l'eau")
    st.write("L'eau est : ")
    
    if prediction[0] == 1:
        st.write("<span class='potability potable'>Potable</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='potability non-potable'>Non-Potable</span>", unsafe_allow_html=True)
        
    st.write("Probabilité d'etre potable : ", model.predict_proba(input_array_scaled)[0][1])
    st.write("Probabilité d'etre non potable: ", model.predict_proba(input_array_scaled)[0][0])
    
    #st.write("This app can assist in evaluating water quality, but should not be used as a substitute for professional analysis.")

# Main function to run the Streamlit app

def main():
    
    st.set_page_config(
        page_title="Water Potability Predictor",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    #with open("assets/style.css") as f:
    #    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    input_data = add_sidebar()
    #st.write(input_data)
    
    with st.container():
        st.title("Prédicteur de potabilité ")
        st.write("Pour la détermination de la qualité de l'eau, veuillez modifier les valeurs des caractéristiques")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)

if __name__ == '__main__':
    main()
