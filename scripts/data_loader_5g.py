import json
from typing import Union, List

import numpy as np
import pandas as pd
import scipy.io as sio


# column names for the dataset in the .mat filee


def flatten_nested_array(nested_array: np.array) -> np.array:
    """
    Flatten a nested array to a single value if it contains only one element.
    """
    if isinstance(nested_array, np.ndarray) and nested_array.size == 1:
        return nested_array.item()
    return nested_array


def get_config(type: str):
    config = json.loads("/config/matab_parsing.json")

    matrix_cols = config[type]["matrix_cols"]
    dataset_cols = config[type]["dataset_cols"]
    dataset_dtypes = config[type]["dataset_dtypes"]

    return matrix_cols, dataset_cols, dataset_dtypes


def parse_matlab_5G(data: list) -> pd.DataFrame:

    matrix_cols, dataset_cols, dataset_dtypes = get_config("5G")

    data_list = [
        {
            "lat": flatten_nested_array(row[0]),
            "lng": flatten_nested_array(row[1]),
            "measurements_matrix": pd.DataFrame(
                row[2], columns=list(matrix_cols.keys())
            ).astype(matrix_cols),
            "num_npcis_rf_op1": flatten_nested_array(row[3]),
            "logical_rf_op1": row[4].flatten(),
            "num_npcis_toa_op1": flatten_nested_array(row[5]),
            "logical_toa_op1": row[6].flatten(),
            "num_npcis_rf_op2": flatten_nested_array(row[7]),
            "logical_rf_op2": row[8].flatten(),
            "num_npcis_toa_op2": flatten_nested_array(row[9]),
            "logical_toa_op2": row[10].flatten(),
            "campaign_id": row[15].flatten()[0],
        }
        for row in data
    ]

    return pd.DataFrame(data_list, columns=dataset_cols).astype(dataset_dtypes)


def load_matlab_file_as_df(
    filename: str, dataset: str, usecols: Union[None, List[str]] = None
) -> pd.DataFrame:
    """
    Load the selected filename from a MATLAB file into a pandas DataFrame.

    :param filename: str, the path to the .mat file.
    :param dataset: str, the name of the dataset to load from the .mat file.
    :param usecols: list of str, the column names to include in the DataFrame.
    :return: pd.DataFrame, the data as a pandas DataFrame.
    :raises ValueError: if the dataset is not found in the MATLAB file.
    """
    # Load the .mat file
    mat_contents = sio.loadmat(filename)

    if dataset not in mat_contents:
        raise ValueError(f"Dataset '{dataset}' not found in MATLAB file.")

    data = mat_contents[dataset]

    data_list = [
        {
            "lat": flatten_nested_array(row[0]),
            "lng": flatten_nested_array(row[1]),
            "measurements_matrix": pd.DataFrame(
                row[2], columns=list(matrix_cols_5G.keys())
            ).astype(matrix_cols_5G),
            "num_npcis_rf_op1": flatten_nested_array(row[3]),
            "logical_rf_op1": row[4].flatten(),
            "num_npcis_toa_op1": flatten_nested_array(row[5]),
            "logical_toa_op1": row[6].flatten(),
            "num_npcis_rf_op2": flatten_nested_array(row[7]),
            "logical_rf_op2": row[8].flatten(),
            "num_npcis_toa_op2": flatten_nested_array(row[9]),
            "logical_toa_op2": row[10].flatten(),
            "campaign_id": row[15].flatten()[0],
        }
        for row in data
    ]

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data_list, columns=dataset_cols).astype(dataset_dtypes)

    # Only include wanted columns
    if usecols is not None:
        df = df[usecols]

    return df


def load_dataframe(filename: str) -> pd.DataFrame:
    """
    Load the selected filename from a MATLAB file into a pandas DataFrame.
    If .h5 file exists, load the data into a pandas DataFrame.

    :param filename: str, the path to the .mat file.
    :return: pd.DataFrame, the data as a pandas DataFrame.
    """
    dataframe_filename = f"{filename}_dataframe.h5"

    try:
        df = pd.read_hdf(dataframe_filename)
        print(f"Loaded dataframe from .h5 file: {dataframe_filename}")
    except FileNotFoundError:
        print(f"Loading data from matlab file: {filename}")

        df = load_matlab_file_as_df(
            filename=filename,
            dataset="dataSet_interp",  # dataSet, dataSet_interp or dataSet_smooth
            usecols=["lat", "lng", "measurements_matrix", "campaign_id"],
        )
        df.to_hdf(dataframe_filename, key="df", mode="w")

    return df
