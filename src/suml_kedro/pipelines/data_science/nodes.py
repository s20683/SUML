import json
import os
from datetime import datetime
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame to split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The training and testing DataFrames.
    """
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df


def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> TabularPredictor:
    """
    Train a model using AutoGluon on the provided training data and evaluate it on the testing data.

    Args:
        train_df (pd.DataFrame): The training DataFrame.
        test_df (pd.DataFrame): The testing DataFrame.

    Returns:
        TabularPredictor: The trained AutoGluon TabularPredictor.
    """
    train_data = TabularDataset(train_df)
    test_data = TabularDataset(test_df)

    # Create folder with current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"data/06_models/model_{current_time}"
    os.makedirs(folder_name)

    # Define label and train AutoGluon model
    label = 'sellingprice'
    predictor = (TabularPredictor(label=label, path=folder_name)
                 .fit(train_data, time_limit=60, presets=['high_quality', 'optimize_for_deployment']))

    # Evaluate model on test data
    metrics = predictor.evaluate(test_data)

    # Save predictor and metrics to folder
    with open(os.path.join(folder_name, "metrics.txt"), "w") as f:
        json.dump(metrics, f)

    return predictor
