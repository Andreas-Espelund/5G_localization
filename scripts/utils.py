from enum import Enum

import numpy as np
import pandas as pd


class RF_PARAM(Enum):
    RSSI = "RSSI"
    NSINR = "NSINR"
    NRSRP = "NRSRP"
    NRSRQ = "NRSRQ"


MISS_REF_VALUES = {
    RF_PARAM.RSSI: -160,
    RF_PARAM.NSINR: -40,
    RF_PARAM.NRSRQ: -40,
    RF_PARAM.NRSRP: -160,
}


def get_miss_ref_value(rf_param: RF_PARAM) -> int:
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


def filter_unique_npcis_by_operator(df, operator_choice):
    # Extract NPCIs and operator IDs from the measurements
    npcis_with_operators = np.concatenate(
        df['measurements_matrix'].apply(lambda x: x[['NPCI', 'operatorID']].values).values)

    # Filter NPCIs based on the operator choice
    filtered_npcis = [npc for npc, op in npcis_with_operators if op in operator_choice]
    unique_npcis_filtered = np.unique(filtered_npcis)

    return unique_npcis_filtered


def extract_unique_npcis(df, operator_choice) -> np.array:
    npcis = []
    for measurements in df['measurements_matrix']:
        for _, row in measurements.iterrows():
            npc = row['NPCI'].astype(int)
            enodeb_id = row['eNodeBID'].astype(int)
            operator_id = row['operatorID'].astype(int)
            # Append as a tuple
            npcis.append((npc, enodeb_id, operator_id))

    # Convert to a DataFrame and drop duplicates
    npcis_df = pd.DataFrame(npcis, columns=['NPCI', 'eNodeBID', 'operatorID'])
    unique_npcis = npcis_df.drop_duplicates()

    # Filter by operator choice
    filtered_npcis = unique_npcis[unique_npcis['operatorID'].isin(operator_choice)]

    return filtered_npcis.to_numpy()


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
    return km
