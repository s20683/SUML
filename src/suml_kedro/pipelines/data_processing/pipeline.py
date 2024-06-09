from kedro.pipeline import Pipeline, node, pipeline
import pandas as pd

from .nodes import process_car_prices, get_car_mapping, get_untied_parameters


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=process_car_prices,
                inputs="car_prices",
                outputs="processed_car_prices",
                name="process_car_prices_node",
            ),
            node(
                func=get_car_mapping,
                inputs="processed_car_prices",
                outputs="car_mapping",
                name="get_car_mapping_node",
            ),
            node(
                func=get_untied_parameters,
                inputs="processed_car_prices",
                outputs=["colors", "interior", "transmission"],
                name="get_untied_parameters_node",
            ),
        ]
    )
