from typing import Tuple, Dict, Any, Set, Optional
import pandas as pd
import re


def _clean_and_parse_datetime(dt_str: Any) -> Optional[int]:
    """
    Clean and parse the year from a datetime string.

    Args:
        dt_str (Any): The datetime string to parse.

    Returns:
        Optional[int]: The extracted year as an integer, or None if no year is found.
    """
    year_pattern = r"[^0-9].* \d\d (\d{4}).*"
    match = re.search(year_pattern, str(dt_str))
    if match:
        year_str = match.group(1)
        year = int(year_str)
        return year
    else:
        return None


def _make_model_trim_mapping(df: pd.DataFrame) -> Dict[str, Dict[str, Set[str]]]:
    """
    Create a mapping of make and model to their respective trim sets.

    Args:
        df (pd.DataFrame): The DataFrame containing 'make', 'model', and 'trim' columns.

    Returns:
        Dict[str, Dict[str, Set[str]]]: A nested dictionary mapping makes to models to trim sets.
    """
    make_model_trim_mapping = df.groupby(['make', 'model'])['trim'].agg(set).reset_index()
    mapping = {}
    for _, row in make_model_trim_mapping.iterrows():
        make = row['make']
        model = row['model']
        trims = row['trim']
        mapping.setdefault(make, {}).setdefault(model, set()).update(trims)
    return mapping


def process_car_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the car prices DataFrame by cleaning and parsing columns.

    Args:
        df (pd.DataFrame): The input DataFrame to process.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    df.dropna(inplace=True)
    df['saleyear'] = df['saledate'].apply(_clean_and_parse_datetime)
    df['years_on_sale'] = (df['saleyear'] - df['year'])
    df.drop(columns=['vin', 'state', 'seller', 'mmr', 'saledate', 'body'], inplace=True)
    return df


def get_car_mapping(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Get a mapping of car makes and models to their trims in JSON-like format.

    Args:
        df (pd.DataFrame): The input DataFrame containing car data.

    Returns:
        Dict[str, Dict[str, Any]]: The car mapping as a nested dictionary.
    """
    mapping = _make_model_trim_mapping(df)
    mapping_json = {make: {model: list(trims) for model, trims in models.items()} for make, models in mapping.items()}
    return mapping_json


def get_untied_parameters(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Get separate DataFrames for car colors, interiors, and transmissions.

    Args:
        df (pd.DataFrame): The input DataFrame containing car data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames for colors, interiors, and transmissions.
    """
    return get_colors(df), get_interior(df), get_transmission(df)


def get_colors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get a DataFrame of unique car colors.

    Args:
        df (pd.DataFrame): The input DataFrame containing car data.

    Returns:
        pd.DataFrame: A DataFrame containing unique car colors.
    """
    unique_values = df['color'].unique()
    return pd.DataFrame({'color': unique_values})


def get_interior(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get a DataFrame of unique car interiors.

    Args:
        df (pd.DataFrame): The input DataFrame containing car data.

    Returns:
        pd.DataFrame: A DataFrame containing unique car interiors.
    """
    unique_values = df['interior'].unique()
    return pd.DataFrame({'interior': unique_values})


def get_transmission(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get a DataFrame of unique car transmissions.

    Args:
        df (pd.DataFrame): The input DataFrame containing car data.

    Returns:
        pd.DataFrame: A DataFrame containing unique car transmissions.
    """
    unique_values = df['transmission'].unique()
    return pd.DataFrame({'transmission': unique_values})
