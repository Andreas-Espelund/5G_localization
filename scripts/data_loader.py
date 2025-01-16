import json
import os
from typing import Union, List

import numpy as np
import pandas as pd
import scipy.io as sio

from scripts.utils import NETWORK_TYPE


# column names for the dataset in the .mat files
# dataset_cols = [
#     "lat",
#     "lng",
#     "measurements_matrix",
#     "num_npcis_rf_op1",
#     "logical_rf_op1",
#     "num_npcis_toa_op1",
#     "logical_toa_op1",
#     "num_npcis_rf_op2",
#     "logical_rf_op2",
#     "num_npcis_toa_op2",
#     "logical_toa_op2",
#     "num_npcis_rf_op3",
#     "logical_rf_op3",
#     "num_npcis_toa_op3",
#     "logical_toa_op3",
#     "campaign_id",
# ]
#
# # specific datatypes for columns
# dataset_dtypes = {
#     "lat": "float64",
#     "lng": "float64",
#     "num_npcis_rf_op1": "int8",
#     "num_npcis_toa_op1": "int8",
#     "num_npcis_rf_op2": "int8",
#     "num_npcis_toa_op2": "int8",
#     "num_npcis_rf_op3": "int8",
#     "num_npcis_toa_op3": "int8",
# }
#
# # column names for the measurement matrix, with datatypes
# matrix_cols = {
#     "NPCI": "int8",
#     "eNodeBID": "int32",
#     "RSSI": "float64",
#     "NSINR": "float64",
#     "NRSRP": "float64",
#     "NRSRQ": "float64",
#     "ToA": "float64",
#     "operatorID": "int8",
#     "campaignID": "int8",
# }
#


def get_config(network_type: NETWORK_TYPE):
    # Construct the full path to the JSON file
    config_path = os.path.join(
        os.path.dirname(__file__), "../config/matlab_parsing.json"
    )

    # Open and load the JSON configuration file
    with open(config_path, "r") as file:
        config = json.load(file)

    # Extract the configuration for the specified network type
    key = network_type.value
    matrix_cols = config[key]["matrix_cols"]
    dataset_cols = config[key]["dataset_cols"]
    dataset_dtypes = config[key]["dataset_dtypes"]

    return matrix_cols, dataset_cols, dataset_dtypes


def flatten_nested_array(nested_array: np.array) -> np.array:
    """
    Flatten a nested array to a single value if it contains only one element.
    """
    if isinstance(nested_array, np.ndarray) and nested_array.size == 1:
        return nested_array.item()
    return nested_array


def parse_matlab_5G(data: list) -> pd.DataFrame:

    matrix_cols, dataset_cols, dataset_dtypes = get_config(NETWORK_TYPE._5G)

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
            "campaign_id": row[11].flatten()[0],
        }
        for row in data
    ]

    return pd.DataFrame(data_list, columns=dataset_cols).astype(dataset_dtypes)


def parse_matlab_NB_IoT(data: list) -> pd.DataFrame:

    matrix_cols, dataset_cols, dataset_dtypes = get_config(NETWORK_TYPE.NB_IoT)

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
            "num_npcis_rf_op3": flatten_nested_array(row[11]),
            "logical_rf_op3": row[12].flatten(),
            "num_npcis_toa_op3": flatten_nested_array(row[13]),
            "logical_toa_op3": row[14].flatten(),
            "campaign_id": row[15].flatten()[0],
        }
        for row in data
    ]

    return pd.DataFrame(data_list, columns=dataset_cols).astype(dataset_dtypes)


def load_matlab_file_as_df(
    filename: str,
    network_type: NETWORK_TYPE,
    dataset: str,
    usecols: Union[None, List[str]] = None,
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

    if network_type == NETWORK_TYPE.NB_IoT:
        df = parse_matlab_NB_IoT(data)
    elif network_type == NETWORK_TYPE._5G:
        df = parse_matlab_5G(data)
    else:
        raise ValueError(f"Network type '{network_type}' not supported.")

    # Only include wanted columns
    if usecols is not None:
        df = df[usecols]

    return df


def load_dataframe(filename: str, network_type: NETWORK_TYPE) -> pd.DataFrame:
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
            network_type=network_type,
            dataset="dataSet_interp",  # dataSet, dataSet_interp or dataSet_smooth
            usecols=["lat", "lng", "measurements_matrix", "campaign_id"],
        )
        df.to_hdf(dataframe_filename, key="df", mode="w")

    return df
