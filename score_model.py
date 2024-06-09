from datetime import datetime
from pathlib import Path

import pandas as pd
import json
import requests
import streamlit as st
from autogluon.tabular import TabularDataset, TabularPredictor
import os

# Path to the directory containing subfolders
models_dir = Path('data/06_models')

# Get a list of subfolder names
subfolders = [subfolder.name for subfolder in models_dir.iterdir() if subfolder.is_dir()]

# Allow the user to choose a subfolder
selected_subfolder = st.sidebar.selectbox('Select model:', subfolders)

selected_subfolder_path = models_dir / selected_subfolder

# Load the trained predictor
predictor = TabularPredictor.load(selected_subfolder_path)

st.sidebar.markdown("# Model Metrics")
metrics_file = os.path.join(selected_subfolder_path, "metrics.txt")
with open(metrics_file, 'r') as file:
    st.sidebar.json(json.load(file))

# Display selected model information
st.sidebar.markdown("# Model Details")
st.sidebar.json(predictor.info()["model_info"][predictor.info()["best_model"]])


# Function to read and process the make-model-trim mapping from a JSON file
def read_make_model_trim_mapping_from_json(file_name):
    with open(file_name, 'r') as file:
        mapping_json = json.load(file)
    # Convert lists back to sets for easier manipulation
    mapping = {make: {model: trims for model, trims in models.items()} for make, models in mapping_json.items()}
    return mapping

# Function to read values from a text file
def read_values_from_txt(file_name):
    with open(file_name, 'r') as file:
        values = file.read().splitlines()
    return values


# Load mappings and color values
mapping = read_make_model_trim_mapping_from_json("data/03_primary/car_mapping.json")
interior_values = read_values_from_txt("data/03_primary/interior.csv")
color_values = read_values_from_txt("data/03_primary/colors.csv")

# Streamlit UI components for user input
st.title("Car Data Input Form")

# Year
year = st.number_input('Year', min_value=1900, max_value=2024, value=2012)

# Make selection
make = st.selectbox('Make', options=list(mapping.keys()))

# Update model options based on selected make
model_options = list(mapping[make].keys())
model = st.selectbox('Model', options=model_options)

# Update trim options based on selected make and model
trim_options = mapping[make][model]
trim = st.selectbox('Trim', options=trim_options)

# Transmission
transmission = st.selectbox('Transmission', ['automatic', 'manual'], index=0)

# Condition
condition = st.number_input('Condition: estimated current state from 1 to 50, where 50 is mint condition.',
                            min_value=1, max_value=50, value=20)

# Odometer
odometer = st.number_input('Odometer', min_value=0, value=100000)

# Color
color = st.selectbox('Color', options=color_values, index=color_values.index('black') if 'black' in color_values else 0)

# Interior
interior = st.selectbox('Interior', options=interior_values, index=interior_values.index('white') if 'white' in interior_values else 0)

# Fetch car image
def fetch_car_image(make, model, year):
    url = f"https://api.carimagery.com/api.asmx/GetImageUrl?searchTerm={year}+{make}+{model}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return None
    except Exception as e:
        return None

# Display the fetched car image
# image_url = fetch_car_image(make, model, year)
# if image_url:
#     st.image(image_url, caption=f"{year} {make} {model}", use_column_width=True)
# else:
#     st.write(f"No image found for the specified car: {year} {make} {model}")

# Function to process input data and predict car price
def predict_car_price():
    current_year = datetime.today().year
    input_data = {
        'year': [year],
        'make': [make],
        'model': [model],
        'trim': [trim],
        'transmission': [transmission],
        'condition': [condition],
        'odometer': [odometer],
        'color': [color],
        'interior': [interior],
        'saleyear': [current_year],
        'years_on_sale': [current_year - year]
    }
    df = pd.DataFrame(input_data)
    tabular_data = TabularDataset(df)
    result = predictor.predict(tabular_data)
    st.write(f"# Predicted Price: ${result.values[0]:.2f} USD")

# Add button to trigger the price prediction
if st.button('Predict Price'):
    predict_car_price()
