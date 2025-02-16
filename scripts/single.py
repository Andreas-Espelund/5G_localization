import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from scripts.utils import haversine_distance, get_miss_ref_value, RF_PARAM_5G


def create_point_vector(
    point: pd.DataFrame, unique_npcis: np.array, rf_param: RF_PARAM_5G
):
    """
    Creates and populates a point vector and valid index vector for a single point.

    :param point: single data point to create the vector from
    :param unique_npcis: all the unique npcis to include in the point vector
    :param rf_param: RF parameter to use
    :return: point vector and idx vector
    """
    unique_npcis = [tuple(row) for row in unique_npcis]
    num_unique_npcis = len(unique_npcis)

    # Initialize the point vector and index vector
    point_vector = np.zeros(num_unique_npcis, dtype=np.float64)

    # Fill with default values
    miss_value = get_miss_ref_value(rf_param)
    point_vector[:] = miss_value

    idx_vector = np.zeros(num_unique_npcis)

    # Map each unique NPCI to its index
    npc_index_map = {npc: idx for idx, npc in enumerate(unique_npcis)}

    measurements = point["measurements_matrix"]
    for _, measurement_row in measurements.iterrows():
        npc_tuple = (
            measurement_row["pci"],
            measurement_row["beam_index"],
            measurement_row["nr_arfcn"],
            measurement_row["operator_id"],
        )
        if npc_tuple in npc_index_map:
            idx = npc_index_map[npc_tuple]
            rf_value = measurement_row[rf_param.value]
            if not np.isnan(rf_value):
                point_vector[idx] = rf_value
                idx_vector[idx] = 1

    return point_vector, idx_vector


def compute_weights_single(
    m_rfp: np.array, idx_rfp: np.array, v_tp: np.array, idx_tp: np.array
) -> (np.array, np.array):
    """
    Computes weights for a single test point against reference points.

    :param m_rfp: point matrix for the reference points (2D array)
    :param idx_rfp: valid index matrix for the reference points (2D array)
    :param v_tp: point vector for the single test point (1D array)
    :param idx_tp: valid index vector for the test point (1D array)
    :return: Weights and sorted indices by weight
    """
    # Reshape test point vector to 2D array with one row
    v_tp_reshaped = v_tp.reshape(1, -1)
    idx_tp_reshaped = idx_tp.reshape(1, -1)

    # Compute the Euclidean distances between the TP and RPs
    D = cdist(v_tp_reshaped, m_rfp, metric="euclidean")

    # Normalize distances based on common valid indices
    match = np.logical_and(idx_tp_reshaped[:, np.newaxis, :], idx_rfp[np.newaxis, :, :])
    s = np.sum(match, axis=2)

    # Avoid division by zero
    realmax = np.finfo(np.float64).max
    D = np.divide(D, s, out=np.full_like(D, realmax), where=s != 0)

    # Set distances to dummy reference points to a very large value
    dummy_rfps = np.all(idx_rfp == 0, axis=1)
    D[:, dummy_rfps] = realmax

    # Replace zero distances with a small value
    min_nonzero_distance = np.min(D[D > 0])
    D[D == 0] = min_nonzero_distance / 20

    # Sort distances and compute weights
    idx_sort = np.argsort(D, axis=1)
    D_sort = np.take_along_axis(D, idx_sort, axis=1)
    W = 1.0 / D_sort

    return W[0], idx_sort[0]  # Return 1D arrays


def wknn_single(
    test_point: pd.Series,
    df_rp: pd.DataFrame,
    idx_sort: np.ndarray[int],
    W: np.ndarray[np.float64],
    k: int,
) -> (np.array, float):
    """
    Compute wKNN for a single test point.

    :param test_point: Single test point as a Series
    :param df_rp: Dataframe of reference points
    :param idx_sort: sorted index array for the test point
    :param W: the weight array for the test point
    :param k: Number of neighbors for wKNN
    :return: Estimated location and error
    """
    # Get real position of test point
    real_lat = test_point["lat"]
    real_long = test_point["lng"]

    # Select the k-nearest reference points
    RFP_selected_idx = idx_sort[:k]

    # Get coordinates of selected reference points
    selected_rps = df_rp.iloc[RFP_selected_idx]
    lat_k_RFP = selected_rps["lat"].values
    long_k_RFP = selected_rps["lng"].values

    # Compute weighted coordinates
    weights_k = W[:k]
    sum_weights = np.sum(weights_k)

    if sum_weights == 0:
        lat_est = np.nan
        long_est = np.nan
    else:
        lat_est = np.sum(lat_k_RFP * weights_k) / sum_weights
        long_est = np.sum(long_k_RFP * weights_k) / sum_weights

    # Compute error using Haversine formula
    error = haversine_distance(real_lat, real_long, lat_est, long_est)

    # Return estimated location and error
    return np.array([lat_est, long_est]), error
