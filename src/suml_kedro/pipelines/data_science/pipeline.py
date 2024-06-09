from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs="processed_car_prices",
                outputs=["train_df", "test_df"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["train_df", "test_df"],
                outputs="regressor",
                name="train_model_node",
            ),
        ]
    )
