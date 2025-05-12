import json
import os
from enum import Enum

import numpy as np
import pandas as pd


class NB_IoT_RF_PARAM(Enum):
    RSSI = "RSSI"
    NSINR = "NSINR"
    NRSRP = "NRSRP"
    NRSRQ = "NRSRQ"


class RF_PARAM_5G(Enum):
    RSSI = "rssi"
    SINR = "sinr"
    RSRP = "rsrp"
    RSRQ = "rsrq"


class NETWORK_TYPE(Enum):
    _5G = "5G"
    NB_IoT = "NB-IoT"


MISS_REF_VALUES = {
    RF_PARAM_5G.RSSI: -160,
    RF_PARAM_5G.SINR: -40,
    RF_PARAM_5G.RSRQ: -40,
    RF_PARAM_5G.RSRP: -160,
}

cluster_color_mapping = {
    0: "orange",
    1: "red",
    2: "green",
    3: "blue",
    4: "yellow",
    5: "purple",
    6: "brown",
    7: "pink",
    8: "teal",
    9: "gray",
    10: "cyan",
    11: "magenta",
    12: "lime",
    13: "navy",
    14: "maroon",
    15: "olive",
    16: "silver",
    17: "gold",
    18: "lavender",
    19: "wheat",
    20: "turquoise",
}


def get_config(filename: str, key: str = None) -> dict:
    config_path = get_abs_filepath(os.path.join("config", filename))
    with open(config_path, "r") as f:
        config = json.load(f)
        if key is None:
            return config
        return config[key]


def get_abs_filepath(path: str) -> str:
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        return os.path.abspath(os.path.join(config["project_root"], path))


def params_to_str(params: list[RF_PARAM_5G]):
    return ",".join(map(lambda x: x.value, params))


def operators_to_str(operators: list[int]):
    return ",".join(map(lambda x: str(x), operators))


def get_miss_ref_value(rf_param: RF_PARAM_5G) -> int:
    """
    Get the default value for missing data
    :param rf_param: The selected RF_PARAM
    :return: value
    """

    return MISS_REF_VALUES[rf_param]


def extract_unique_npcis(measurements: pd.Series) -> list:
    # Concatenate all measurements matrices into a single DataFrame
    all_measurements = pd.concat(measurements.tolist(), ignore_index=True)

    cols = ["pci", "beam_index", "nr_arfcn", "operator_id"]
    all_measurements = all_measurements[cols]

    # Drop duplicates based on the specified columns
    unique_npcis = all_measurements.drop_duplicates(subset=cols)

    # Convert the DataFrame to a list of tuples
    return list(unique_npcis.itertuples(index=False, name=None))


def haversine_distance(lat1, lon1, lat2, lon2) -> tuple[float, float, float]:
    """
    Calculate the great circle distance between two points

    :param lat1: point a latitude
    :param lon1: point a longitude
    :param lat2: point b latitude
    :param lon2: point b longitude
    :return: distance in km, nmi and mi
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of Earth in kilometers (mean radius)
    r = 6371.0
    km = c * r
    m = km * 1000
    nmi = km * 0.539956803  # nautical miles
    mi = km * 0.621371192  # miles
    return m


def dataset_tp_rp_split(
    df: pd.DataFrame, test_point_probability: float, random_seed: int
) -> (pd.DataFrame, pd.DataFrame):
    """
    Split the dataset into Test points TPs and Reference points RPs

    :param df: Original dataset
    :param test_point_probability: Probability of a point beeing a test point
    :param random_seed: Random seed
    :return: test points and reference points
    """
    np.random.seed(random_seed)
    test_mask = np.random.rand(len(df)) <= test_point_probability
    df_tp = df[test_mask].reset_index(drop=True)
    df_rp = df[~test_mask].reset_index(drop=True)
    return df_tp, df_rp


def get_arfcns_from_bands(bands: list[int]) -> list[int]:
    band_config = get_config("band_map.json")
    return [
        int(arfcn)
        for mapping in band_config.values()
        for arfcn, band in mapping.items()
        if band in bands
    ]
