from typing import Dict

from autogluon.tabular import TabularPredictor
import pandas as pd
from matplotlib import pyplot as plt


def evaluate_model(predictor: TabularPredictor, test_data: pd.DataFrame) -> dict:
    """
    Evaluate the model on the test data and generate predictions.

    Args:
        predictor (TabularPredictor): The trained AutoGluon TabularPredictor.
        test_data (pd.DataFrame): The testing DataFrame.

    Returns:
        dict: A dictionary containing the model's performance metrics and predictions.
    """
    # Evaluate the model
    performance = predictor.evaluate(test_data)

    # Generate predictions
    predictions = predictor.predict(test_data)

    # Prepare a report
    report = {
        "performance": performance,
        "predictions": predictions.to_list(),
    }

    return report


def calculate_feature_importance(predictor: TabularPredictor, train_data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate the feature importance scores from the trained predictor.

    Args:
        predictor (TabularPredictor): The trained AutoGluon TabularPredictor.
        train_data (pd.DataFrame): The training DataFrame.

    Returns:
        Dict[str, float]: A dictionary mapping feature names to their importance scores.
    """
    importance_df = predictor.feature_importance(train_data)
    feature_importance_dict = dict(zip(importance_df.index, importance_df['importance']))
    return feature_importance_dict


def plot_feature_importance(feature_importance: Dict[str, float]) -> None:
    """
    Plot and save the feature importance scores.

    Args:
        feature_importance (Dict[str, float]): A dictionary mapping feature names to their importance scores.
    """
    # Extract feature names and importance scores from the dictionary
    feature_names = list(feature_importance.keys())
    importance_scores = list(feature_importance.values())

    # Plot the feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, importance_scores)
    plt.xticks(rotation=45, ha='right')  # Rotate x labels for better readability
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig('data/08_reporting/feature_importance_plot.png')
    plt.close()
