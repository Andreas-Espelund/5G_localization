import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from scripts.data_processing import cluster_data_and_train_random_forest
from scripts.utils import (
    RF_PARAM,
    get_miss_ref_value,
    haversine_distance,
    dataset_tp_rp_split,
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
        shape=[num_points, num_unique_npcis],
        fill_value=miss_ref_value,
        dtype=np.float64,
    )

    idx_matrix = np.zeros(shape=[num_points, num_unique_npcis])
    npc_index_map = {npc: idx for idx, npc in enumerate(unique_npcis)}

    i = 0
    for _, row in df.iterrows():
        measurements = row["measurements_matrix"]
        for _, row in measurements.iterrows():
            npc_tuple = (row["NPCI"], row["eNodeBID"], row["operatorID"])
            rf_value = row[rf_param.value]
            # Check if the tuple is in the unique_npcis
            if npc_tuple in npc_index_map:
                idx = npc_index_map[npc_tuple]
                if not np.isnan(rf_value):
                    point_matrix[i, idx] = rf_value
                    idx_matrix[i, idx] = 1
        i += 1
    return point_matrix, idx_matrix


def compute_weights(
    m_rfp: np.array, idx_rfp: np.array, m_tp: np.array, idx_tp: np.array
) -> (np.array, np.array):
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
    k_avg_error = []

    # Extract real positions of test points
    real_lat = df_tp["lat"].values
    real_long = df_tp["lng"].values
    real_position = np.vstack((real_lat, real_long)).T

    # Loop over each k value
    for i, this_k in enumerate(k_values):
        # Select the k-nearest reference points
        RFP_selected_idx = idx_sort[:, :this_k]

        df_tp["nearest"] = RFP_selected_idx

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
        try:
            lat_k_TP = np.where(sum_weights != 0, sum_lat / sum_weights, np.nan)
            long_k_TP = np.where(sum_weights != 0, sum_long / sum_weights, np.nan)
        except ZeroDivisionError:
            lat_k_TP = np.nan
            long_k_TP = np.nan
            print(f"zero devision error for {this_k}")

        # Compute errors using Haversine formula
        km_pow = haversine_distance(
            real_position[:, 0], real_position[:, 1], lat_k_TP, long_k_TP
        )
        average_error_pow = np.mean(km_pow)

        # k_avg_error[this_k] = average_error_pow
        k_avg_error.append(average_error_pow)
        # Store estimated locations
        TP_est_location_k = np.zeros((num_tps, 2))
        TP_est_location_k[:, 0] = lat_k_TP
        TP_est_location_k[:, 1] = long_k_TP
        TP_est_location[i] = TP_est_location_k

    return TP_est_location, np.array(k_avg_error)


def wknn_one(
    df_tp: pd.DataFrame,
    df_rp: pd.DataFrame,
    idx_sort: np.array,
    W: np.array,
    k: int,
) -> (np.array, float):
    """
    :param df_tp: Dataframe of reference points
    :param df_rp: Dataframe of test points
    :param idx_sort: sorted index matrix by weights
    :param W: the weight matrix for the test/reference points
    :param k: Number of neighbors for wKNN
    :return: Estimated locations and average error for the given k value
    """
    num_tps = df_tp.shape[0]
    selected_rps_per_tp = {}

    # Extract real positions of test points
    real_lat = df_tp["lat"].values
    real_long = df_tp["lng"].values
    real_position = np.vstack((real_lat, real_long)).T

    # Select the k-nearest reference points
    RFP_selected_idx = idx_sort[:, :k]

    # Extract coordinates of the selected reference points
    lat_k_RFP_matrix = df_rp.iloc[RFP_selected_idx.flatten()]["lat"].values.reshape(
        RFP_selected_idx.shape
    )
    long_k_RFP_matrix = df_rp.iloc[RFP_selected_idx.flatten()]["lng"].values.reshape(
        RFP_selected_idx.shape
    )

    # Compute weighted sums of coordinates
    sum_lat = np.sum(lat_k_RFP_matrix * W[:, :k], axis=1)
    sum_long = np.sum(long_k_RFP_matrix * W[:, :k], axis=1)

    # Compute estimated coordinates of test points
    sum_weights = np.sum(W[:, :k], axis=1)
    lat_k_TP = np.where(sum_weights != 0, sum_lat / sum_weights, np.nan)
    long_k_TP = np.where(sum_weights != 0, sum_long / sum_weights, np.nan)

    # Compute errors using Haversine formula
    errors = haversine_distance(
        real_position[:, 0], real_position[:, 1], lat_k_TP, long_k_TP
    )

    # Store estimated locations
    TP_est_location = np.zeros((num_tps, 2))
    TP_est_location[:, 0] = lat_k_TP
    TP_est_location[:, 1] = long_k_TP

    return (
        TP_est_location,
        errors,
    )


def run_weighted_coverage(
    df: pd.DataFrame,
    rf_param: RF_PARAM,
    k_max: int,
    unique_npcis: np.array,
    random_seed: int,
    n_clusters: int,
) -> (np.array, np.array, int, float):

    tmp = df.copy().sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df_tp, df_rp = dataset_tp_rp_split(tmp, 0.3, random_seed)

    if not n_clusters > 0:
        start_time = time.time()
        TP_est_location, k_avg_error = process_test_points(
            df_tp, df_rp, unique_npcis, rf_param, k_max
        )
        end_time = time.time()
        complexity = len(df_tp) * len(df_rp)
        return TP_est_location, k_avg_error, complexity, end_time - start_time

    rf_model = cluster_data_and_train_random_forest(
        df_rp, n_clusters, unique_npcis, rf_param, random_seed
    )

    start_time = time.time()  # dont include model training in the online stage timing
    TP_est_location, k_avg_error, rp_factor = process_clusters(
        df_tp, df_rp, unique_npcis, rf_param, k_max, rf_model
    )
    end_time = time.time()

    return TP_est_location, k_avg_error, rp_factor, end_time - start_time


def process_clusters(df_tp, df_rp, unique_npcis, rf_param, k_max, rf_model):
    # Predict clusters for all test points at once
    tp_features, _ = create_point_matrix(df_tp, unique_npcis, rf_param)
    test_clusters = rf_model.predict(tp_features)
    # Organize test points by cluster
    df_tp["predicted_cluster"] = test_clusters
    cluster_groups = df_tp.groupby("predicted_cluster")

    # Initialize lists to store results
    all_tp_est_locations = []
    all_k_avg_errors = []
    total_rps = 0

    for cluster, group in cluster_groups:
        rps = df_rp[df_rp["cluster"] == cluster]
        total_rps += len(group) * len(rps)

        # Process each cluster's test points
        TP_est_location, k_avg_error = process_test_points(
            group, rps, unique_npcis, rf_param, k_max
        )

        # Store results
        all_tp_est_locations.extend(TP_est_location)
        all_k_avg_errors.extend(k_avg_error)

    # Concatenate all estimated locations and average errors
    TP_est_location_combined = np.vstack(all_tp_est_locations)
    all_k_avg_errors = np.array(all_k_avg_errors)
    return TP_est_location_combined, all_k_avg_errors, total_rps


def process_test_points(df_tp, df_rp, unique_npcis, rf_param, k_max):
    # Create the point matrix for the reference points
    m_rfp, idx_rfp = create_point_matrix(df_rp, unique_npcis, rf_param)

    # Create the point matrix for the test points
    m_tp, idx_tp = create_point_matrix(df_tp, unique_npcis, rf_param)

    # Compute the weights between the test points and reference points
    W, idx_sort = compute_weights(m_rfp, idx_rfp, m_tp, idx_tp)

    # Do wKNN to estimate the positions and errors
    TP_est_location, k_avg_error = wknn_one(df_tp, df_rp, idx_sort, W, k_max)

    return TP_est_location, k_avg_error
