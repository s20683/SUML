from datetime import datetime
import pandas as pd
import json
import requests
import streamlit as st
from autogluon.tabular import TabularDataset, TabularPredictor

# Load the trained predictor
predictor = TabularPredictor.load("AutogluonModels/ag-20240430_144147")

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
mapping = read_make_model_trim_mapping_from_json("make_model_trim.txt")
interior_values = read_values_from_txt("interior_values.txt")
color_values = read_values_from_txt("color_values.txt")

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
    st.write(f"Predicted Price: ${result.values[0]:.2f} USD")

# Add button to trigger the price prediction
if st.button('Predict Price'):
    predict_car_price()
