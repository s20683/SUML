import json
import os
from datetime import datetime
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Split the DataFrame into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df


def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> TabularPredictor:
    # Convert DataFrames to TabularDatasets
    train_data = TabularDataset(train_df)
    test_data = TabularDataset(test_df)

    # Create folder with current name and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"data/06_models/model_{current_time}"
    os.makedirs(folder_name)


    # Define label and train AutoGluon model
    label = 'sellingprice'
    predictor = (TabularPredictor(label=label, path=folder_name)
                 .fit(train_data, time_limit=60, presets=['high_quality', 'optimize_for_deployment']))
    #predictor.delete_models(models_to_keep='best', dry_run=False)

    # Evaluate model on test data
    metrics = predictor.evaluate(test_data)


    # Save predictor and metrics to folder
    with open(os.path.join(folder_name, "metrics.txt"), "w") as f:
        json.dump(metrics, f)

    return predictor




# save_mapping_to_json(make_model_trim_mapping(df), "make_model_trim.txt")
#
# print(df.columns.tolist())
#
# # Get columns with object or category data types
# #categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
# for c in ['color', 'interior', 'transmission']:
#     save_unique_values_to_file(df, c)
#
#
# print(train_data[label].describe())
#
#
#
# y_pred = predictor.predict(test_data.drop(columns=[label]))
# print(y_pred.head())
#
# print(predictor.evaluate(test_data, silent=True))
#
# print("***")
# print(predictor.leaderboard(test_data))
