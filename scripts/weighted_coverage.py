import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from scripts.utils import (
    RF_PARAM,
    get_miss_ref_value,
    haversine_distance
)


def create_point_matrix(df: pd.DataFrame, unique_npcis: np.array, rf_param: RF_PARAM):
    """
    Creates and populates a point matrix and valid index matrix for the test or reference points.
    The point matrix is populated with the given RF value or the miss_ref value.
    The idx matrix is populated with 1 or 0 based on if the point is valid.

    :param df: data points to create the matrix from
    :param unique_npcis: all the unique npcis to include in the point matrix
    :param rf_param: what param to use
    :return: point matrix and idx matrix
    """
    num_points = df.shape[0]
    unique_npcis = [tuple(row) for row in unique_npcis]
    num_unique_npcis = len(unique_npcis)
    miss_ref_value = get_miss_ref_value(rf_param)

    point_matrix = np.full(
        shape=[num_points, num_unique_npcis], fill_value=miss_ref_value
    )

    idx_matrix = np.zeros(shape=[num_points, num_unique_npcis])
    npc_index_map = {npc: idx for idx, npc in enumerate(unique_npcis)}

    for i in range(num_points):
        measurements = df.iloc[i]["measurements_matrix"]
        for _, row in measurements.iterrows():
            npc_tuple = (row["NPCI"], row["eNodeBID"], row["operatorID"])
            rf_value = row[rf_param.value]
            # Check if the tuple is in the unique_npcis
            if npc_tuple in npc_index_map:
                idx = npc_index_map[npc_tuple]
                if not np.isnan(rf_value):
                    point_matrix[i, idx] = rf_value
                    idx_matrix[i, idx] = 1

    return point_matrix, idx_matrix


def compute_weights(m_rfp: np.array, idx_rfp: np.array, m_tp: np.array, idx_tp: np.array) -> (np.array, np.array):
    """
    Computes weights for two matrices.
    This is used for the weights in wKNN.
    :param m_rfp: point matrix for the reference points
    :param idx_rfp: valid index matrix for the reference points
    :param m_tp: point matrix for the test points
    :param idx_tp: valid index matrix for the test points
    :return: Weights and sorted indices by weight
    """
    # Calculate distances between test points and reference points
    D = cdist(m_tp, m_rfp, metric="euclidean")

    # Normalize distances based on common NPCIs
    match = np.logical_and(idx_tp[:, np.newaxis, :], idx_rfp[np.newaxis, :, :])
    s = np.sum(match, axis=2)

    # Avoid division by zero by setting distances to infinity where no matches exist
    D = np.divide(D, s, out=np.full_like(D, np.inf), where=s != 0)

    # Set distances to dummy reference points to a very large value
    dummy_rfps = np.all(idx_rfp == 0, axis=1)
    D[:, dummy_rfps] = np.inf

    # Replace zero distances with a small value to avoid singularities
    min_nonzero_distance = np.min(D[D > 0])
    D[D == 0] = min_nonzero_distance / 20

    # Sort distances and compute weights
    idx_sort = np.argsort(D, axis=1)
    D_sort = np.take_along_axis(D, idx_sort, axis=1)
    W = 1.0 / D_sort

    return W, idx_sort


def wknn(
        df_tp: pd.DataFrame,
        df_rp: pd.DataFrame,
        idx_sort: np.array,
        W: np.array,
        k_max: int,
) -> (np.array, dict):
    """

    :param df_tp: Dataframe of reference points
    :param df_rp: Dataframe of test points
    :param idx_sort: sorted index matrix by weights
    :param W: the weight matrix for the test/reference points
    :param k_max: Max number of neighbors for wKNN
    :return: Estimated locations and average error for each k value
    """
    num_tps = df_tp.shape[0]
    k_values = range(1, k_max + 1)
    TP_est_location = [None] * len(k_values)
    k_avg_error = {}

    # Extract real positions of test points
    real_lat = df_tp["lat"].values
    real_long = df_tp["lng"].values
    real_position = np.vstack((real_lat, real_long)).T

    # Loop over each k value
    for i, this_k in enumerate(k_values):
        # Select the k-nearest reference points
        RFP_selected_idx = idx_sort[:, :this_k]

        # Extract coordinates of the selected reference points
        lat_k_RFP_matrix = df_rp.iloc[RFP_selected_idx.flatten()]["lat"].values.reshape(
            RFP_selected_idx.shape
        )
        long_k_RFP_matrix = df_rp.iloc[RFP_selected_idx.flatten()][
            "lng"
        ].values.reshape(RFP_selected_idx.shape)

        # Compute weighted sums of coordinates
        sum_lat = np.sum(lat_k_RFP_matrix * W[:, :this_k], axis=1)
        sum_long = np.sum(long_k_RFP_matrix * W[:, :this_k], axis=1)

        # Compute estimated coordinates of test points
        sum_weights = np.sum(W[:, :this_k], axis=1)
        lat_k_TP = np.where(sum_weights != 0, sum_lat / sum_weights, np.nan)
        long_k_TP = np.where(sum_weights != 0, sum_long / sum_weights, np.nan)

        # Compute estimated coordinates of test points
        # lat_k_TP = sum_lat / np.sum(W[:, :this_k], axis=1)
        # long_k_TP = sum_long / np.sum(W[:, :this_k], axis=1)

        # Compute errors using Haversine formula
        km_pow = haversine_distance(
            real_position[:, 0], real_position[:, 1], lat_k_TP, long_k_TP
        )
        average_error_pow = np.mean(km_pow)

        k_avg_error[this_k] = average_error_pow

        # Store estimated locations
        TP_est_location_k = np.zeros((num_tps, 2))
        TP_est_location_k[:, 0] = lat_k_TP
        TP_est_location_k[:, 1] = long_k_TP
        TP_est_location[i] = TP_est_location_k

    return TP_est_location, k_avg_error


def run_weighted_coverage(dataset: pd.DataFrame, rf_param: RF_PARAM, k_max: int, unique_npcis: np.array) -> (
        float, float):
    """
    'Main' entry point.
    Splits the dataset into test and reference points.
    Creates the point matrecies and calculates the weights.
    Runs wKNN to estimate position and calculate error.
    :param unique_npcis:
    :param operator_choice:
    :param dataset: Original dataset
    :param rf_param: What rf param to use
    :param k_max: Max number of neighbors for wKNN
    :param random_seed: Random seed for shuffling the dataframe
    :return: Estimated locations and average error for each k value
    """

    # Start timing for copying dataset
    # start_time = time.time()
    # Copy dataset to avoid overwriting
    dataset = dataset.copy()
    # print(f"Copying dataset: {time.time() - start_time:.6f} seconds")

    # Start timing for shuffling dataset
    # start_time = time.time()
    # Shuffle the dataframe
    dataset = dataset.sample(frac=1, random_state=int(time.time())).reset_index(drop=True)
    # print(f"Shuffling dataset: {time.time() - start_time:.6f} seconds")

    # Start timing for splitting dataset
    # start_time = time.time()
    # Randomly assign points as test points (2) or reference points (1)
    test_mask = np.random.rand(len(dataset)) <= 0.3
    df_tp = dataset[test_mask]
    df_rp = dataset[~test_mask]
    # print(f"Splitting dataset: {time.time() - start_time:.6f} seconds")

    # Start timing for creating reference point matrix
    # start_time = time.time()
    # Create matrices for test and reference points
    m_rfp, idx_rfp = create_point_matrix(df_rp, unique_npcis, rf_param)
    # print(f"Creating reference point matrix: {time.time() - start_time:.6f} seconds")

    # Start timing for creating test point matrix
    # start_time = time.time()
    m_tp, idx_tp = create_point_matrix(df_tp, unique_npcis, rf_param)
    # print(f"Creating test point matrix: {time.time() - start_time:.6f} seconds")

    # Start timing for computing weights
    # start_time = time.time()
    W, idx_sort = compute_weights(m_rfp, idx_rfp, m_tp, idx_tp)
    # print(f"Computing weights: {time.time() - start_time:.6f} seconds")

    # Start timing for wKNN computation
    # start_time = time.time()
    tp_est_location, k_avg_error = wknn(df_tp, df_rp, idx_sort, W, k_max)
    # print(f"wKNN computation: {time.time() - start_time:.6f} seconds")

    return tp_est_location, k_avg_error
