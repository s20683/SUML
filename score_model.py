import pandas as pd
import json
from autogluon.tabular import TabularDataset, TabularPredictor

predictor = TabularPredictor.load("AutogluonModels/ag-20240430_144147")


def read_make_model_trim_mapping_from_json(file_name):
    with open(file_name, 'r') as file:
        mapping_json = json.load(file)

    # Convert lists back to sets
    mapping = {make: {model: set(trims) for model, trims in models.items()} for make, models in mapping_json.items()}

    return mapping

mapping = read_make_model_trim_mapping_from_json("make_model_trim.txt")

print(mapping)

# Define the data for the DataFrame
data = {
    'year': [2012],
    'make': ["Kia"],
    'model': ["Soul"],
    'trim': ["LX"],
#    'body': ["SUV"],
    'transmission': ["automatic"],
    'condition': [20],
    'odometer': [100000],
    'color': ["black"],
    'interior': ["white"],
    'saleyear': [2024],
    'years_on_sale': [12]
}

# Create the DataFrame
df = pd.DataFrame(data)

tabular_data = TabularDataset(df)

print(predictor.predict(tabular_data))