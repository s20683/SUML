from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, calculate_feature_importance, plot_feature_importance


def create_pipeline(**kwargs) -> Pipeline:
    """This is a simple pipeline which generates a pair of plots"""
    return pipeline(
        [
            node(
                func=evaluate_model,
                inputs=["regressor", "test_df"],
                outputs="performance_report",
                name="evaluate_model_node",
            ),
            node(
                func=calculate_feature_importance,
                inputs=["regressor", "train_df"],
                outputs="feature_importance_report",
                name="calculate_feature_importance_node",
            ),
            node(
                func=plot_feature_importance,
                inputs="feature_importance_report",
                outputs=None,
                name="plot_feature_importance_node",
            ),
        ]
    )
