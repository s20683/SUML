import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor
import re
import json

def clean_and_parse_datetime(dt_str):
    # Define a regex pattern to match the year (four digits) in the datetime string
    year_pattern = r"[^0-9].* \d\d (\d{4}).*"

    # Use regex to find the year in the datetime string
    match = re.search(year_pattern, str(dt_str))

    if match:
        year_str = match.group(1)  # Extract the matched year as a string
        year = int(year_str)  # Convert the extracted year string to an integer
        return year
    else:
        return None  # Return NaT (Not a Time) if no year is found in the datetime string

def save_unique_values_to_file(df, column_name):
    # Show all unique values in the specified column
    unique_values = df[column_name].unique()

    # Save unique values to a text file
    with open(column_name + "_values.txt", 'w') as file:
        for value in unique_values:
            file.write(str(value) + '\n')

def make_model_trim_mapping(df):
    # Group the DataFrame by 'make' and 'model', and aggregate 'trim' values into a set
    make_model_trim_mapping = df.groupby(['make', 'model'])['trim'].agg(set).reset_index()

    # Convert the grouped data into a dictionary
    mapping = {}
    for _, row in make_model_trim_mapping.iterrows():
        make = row['make']
        model = row['model']
        trims = row['trim']
        mapping.setdefault(make, {}).setdefault(model, set()).update(trims)

    return mapping

def save_mapping_to_json(mapping, file_name):
    # Convert sets to lists for JSON serialization
    mapping_json = {make: {model: list(trims) for model, trims in models.items()} for make, models in mapping.items()}
    with open(file_name, 'w') as file:
        json.dump(mapping_json, file, indent=4)


warnings.filterwarnings('ignore')
np.random.seed(123)

file_path = 'car_prices.csv'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)

df.dropna(inplace=True)

#df = df.sample(n=500000, random_state=42)

# Calculate value counts of car models
model_counts = df['model'].value_counts()

# Clean and parse 'saledate' column
df['saleyear'] = df['saledate'].apply(clean_and_parse_datetime)

# Calculate the difference in years between 'saledate' and 'Year'
df['years_on_sale'] = (df['saleyear'] - df['year'])

df.drop(columns=['vin', 'state', 'seller', 'mmr', 'saledate', 'body'], inplace=True)

# Split the DataFrame into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_data = TabularDataset(train_df)
test_data = TabularDataset(test_df)

save_mapping_to_json(make_model_trim_mapping(df), "make_model_trim.txt")

print(df.columns.tolist())

# Get columns with object or category data types
#categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
for c in ['color', 'interior', 'transmission']:
    save_unique_values_to_file(df, c)

label = 'sellingprice'
print(train_data[label].describe())

predictor = (TabularPredictor(label=label)
             .fit(train_data,  time_limit=600, presets=['high_quality', 'optimize_for_deployment']))
predictor.delete_models(models_to_keep='best', dry_run=False)

y_pred = predictor.predict(test_data.drop(columns=[label]))
print(y_pred.head())

print(predictor.evaluate(test_data, silent=True))

print("***")
print(predictor.leaderboard(test_data))
