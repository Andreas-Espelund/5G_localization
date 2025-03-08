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
    DUMMY = "dummy"


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


# def get_color_map(key: str) -> dict[str, str]:
#     config_path = get_abs_filepath('')
#     with open()


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


def make_filename_details(
    wKNN_value: int,
    wKNN_rf_params: [RF_PARAM_5G],
    cluster_range: [int],
    cluster_params: [RF_PARAM_5G],
    operator_choice: np.array,
    n_runs: int,
):
    return f"_operators[{operators_to_str(operator_choice)}]_runs[{n_runs}]\
_wKNN[K={wKNN_value},RF={params_to_str(wKNN_rf_params)}]\
_clustering[N={cluster_range[0]}-{cluster_range[-1]},RF={params_to_str(cluster_params)}]"


def get_miss_ref_value(rf_param: RF_PARAM_5G) -> int:
    """
    Get the default value for missing data
    :param rf_param: The selected RF_PARAM
    :return: value
    """

    return MISS_REF_VALUES[rf_param]


def dataset_reference_test_split(
    df: pd.DataFrame, test_point_probability: float
) -> (pd.DataFrame, pd.DataFrame):
    """
    Takes the dataset and returns two dataframes for test-points and reference-points
    :param df: Original dataset
    :param test_point_probability: Probability of a point beeing a test point
    :return:
    """

    df["PointType"] = (np.random.rand(len(df)) <= test_point_probability).astype(
        int
    ) + 1
    df_rp = df[df["PointType"] == 1]
    df_tp = df[df["PointType"] == 2]

    return df_tp, df_rp


def replace_nr_arfcns(df: pd.DataFrame, map: dict[int, int]):
    df["measurements_matrix"] = df["measurements_matrix"].apply(
        lambda matrix: matrix.assign(nr_arfcn=matrix["nr_arfcn"].replace(map))
    )


def filter_unique_npcis_by_operator(df, operator_choice):
    # Extract NPCIs and operator IDs from the measurements
    npcis_with_operators = np.concatenate(
        df["measurements_matrix"]
        .apply(lambda x: x[["pci", "operator_id"]].values)
        .values
    )

    # Filter NPCIs based on the operator choice
    filtered_npcis = [npc for npc, op in npcis_with_operators if op in operator_choice]
    unique_npcis_filtered = np.unique(filtered_npcis)

    return unique_npcis_filtered


def extract_unique_npcis_NB_IoT(df, operator_choice) -> np.array:
    npcis = []
    for measurements in df["measurements_matrix"]:
        for _, row in measurements.iterrows():
            npc = row["NPCI"].astype(int)
            enodeb_id = row["eNodeBID"].astype(int)
            operator_id = row["operatorID"].astype(int)
            # Append as a tuple
            npcis.append((npc, enodeb_id, operator_id))

    # Convert to a DataFrame and drop duplicates
    npcis_df = pd.DataFrame(npcis, columns=["NPCI", "eNodeBID", "operatorID"])
    unique_npcis = npcis_df.drop_duplicates()

    # Filter by operator choice
    filtered_npcis = unique_npcis[unique_npcis["operatorID"].isin(operator_choice)]

    return filtered_npcis.to_numpy()


# def extract_unique_npcis(df, operator_choice) -> np.array:
#     npcis = []
#     for measurements in df["measurements_matrix"]:
#         for _, row in measurements.iterrows():
#             npc = row["PCI"].astype(int)
#             enodeb_id = row["SSB_Index"].astype(int)
#             operator_id = row["operatorID"].astype(int)
#             # Append as a tuple
#             npcis.append((npc, enodeb_id, operator_id))
#
#     # Convert to a DataFrame and drop duplicates
#     npcis_df = pd.DataFrame(npcis, columns=["PCI", "SSB_Index", "operatorID"])
#     unique_npcis = npcis_df.drop_duplicates()
#
#     # Filter by operator choice
#     filtered_npcis = unique_npcis[unique_npcis["operatorID"].isin(operator_choice)]
#
#     return filtered_npcis.to_numpy()


def extract_unique_npcis(measurements: pd.Series) -> list:
    # Concatenate all measurements matrices into a single DataFrame
    all_measurements = pd.concat(measurements.tolist(), ignore_index=True)

    cols = ["pci", "beam_index", "nr_arfcn", "operator_id"]
    all_measurements = all_measurements[cols]

    # Drop duplicates based on the specified columns
    unique_npcis = all_measurements.drop_duplicates(subset=cols)

    # Convert the DataFrame to a list of tuples
    return list(unique_npcis.itertuples(index=False, name=None))


def compute_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a dataframe containing key metrics from the results
    :param results: The raw data collected
    :return: a new dataframe with key metrics
    """
    data = []
    columns = [
        "k-value",
        "mean_error",
        "median_error",
        "min_error",
        "max_error",
        "std_dev",
        "mse",
    ]
    for col in results.columns:
        mean = results[col].mean()
        median = results[col].median()
        max = results[col].max()
        min = results[col].min()
        std_dev = results[col].std()
        mse = (results[col] ** 2).mean()
        data.append([f"k={col}", mean, median, min, max, std_dev, mse])

    return pd.DataFrame(data, columns=columns)


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


def create_df_index_map(df: pd.DataFrame) -> dict[int, int]:
    """
    Creates mapping between
    :param df:
    :return:
    """
    return dict(zip(df.index, range(len(df))))


def apply_index_map(index: list, mapping: dict) -> list:
    return [mapping[i] for i in index if i in mapping]
