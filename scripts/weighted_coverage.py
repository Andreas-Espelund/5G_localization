import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.svm import SVC

from scripts.utils import (
    RF_PARAM,
    get_miss_ref_value,
    haversine_distance, train_kmeans
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
        dtype=np.float64
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
    k_avg_error = []

    # Extract real positions of test points
    real_lat = df_tp["lat"].values
    real_long = df_tp["lng"].values
    real_position = np.vstack((real_lat, real_long)).T

    # Loop over each k value
    for i, this_k in enumerate(k_values):
        # Select the k-nearest reference points
        RFP_selected_idx = idx_sort[:, :this_k]

        # Extract coordinates of the selected reference points
        lat_k_RFP_matrix = (df_rp.iloc[RFP_selected_idx.flatten()]["lat"]
                            .values.reshape(RFP_selected_idx.shape))
        long_k_RFP_matrix = (df_rp.iloc[RFP_selected_idx.flatten()]["lng"]
                             .values.reshape(RFP_selected_idx.shape))

        # Compute weighted sums of coordinates
        sum_lat = np.sum(lat_k_RFP_matrix * W[:, :this_k], axis=1)
        sum_long = np.sum(long_k_RFP_matrix * W[:, :this_k], axis=1)

        # Compute estimated coordinates of test points
        sum_weights = np.sum(W[:, :this_k], axis=1)
        lat_k_TP = np.where(sum_weights != 0, sum_lat / sum_weights, np.nan)
        long_k_TP = np.where(sum_weights != 0, sum_long / sum_weights, np.nan)

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


def run_weighted_coverage(dataset: pd.DataFrame, rf_param: RF_PARAM, k_max: int,
                          unique_npcis: np.array, random_seed: int, n_clusters: int, use_svm: bool) -> (float, float):
    """
    'Main' entry point.
    Splits the dataset into test and reference points.
    Creates the point matrecies and calculates the weights.
    Runs wKNN to estimate position and calculate error.
    :param unique_npcis:
    :param dataset: Original dataset
    :param rf_param: What rf param to use
    :param k_max: Max number of neighbors for wKNN

    :return: Estimated locations and average error for each k value
    """
    # Shuffle the dataframe
    start_time = time.time()
    df = dataset.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Randomly assign points as test points (2) or reference points (1)
    np.random.seed(random_seed)
    test_mask = np.random.rand(len(df)) <= 0.3
    df_tp = df[test_mask].copy()
    df_rp = df[~test_mask].copy()

    # run without clustering
    if n_clusters == 0:
        TP_est_location, k_avg_error = process_test_points(df_tp, df_rp, unique_npcis, rf_param, k_max)
        end_time = time.time()
        complexity = len(df_tp) * len(df_rp)
        return TP_est_location, k_avg_error, complexity, end_time - start_time

    # cluster the reference points
    kMeans, cluster_labels = train_kmeans(df_rp, n_clusters, 42)
    df_rp['cluster'] = cluster_labels

    svm = None
    if use_svm:
        # train the SVM model
        svm = SVC(kernel='rbf', gamma=1, C=100, random_state=random_seed)
        svm.fit(df_rp[['lat', 'lng']], df_rp['cluster'])

    TP_est_location, k_avg_error, rp_factor = (
        process_clusters(df_tp, df_rp, kMeans, unique_npcis, rf_param, k_max, svm)
    )
    end_time = time.time()
    k_avg_error = k_avg_error.mean(axis=0)
    # return estimated locations and average error for each k-value
    return TP_est_location, k_avg_error, rp_factor, end_time - start_time


def process_clusters(df_tp, df_rp, kMeans, unique_npcis, rf_param, k_max, svm):
    # Predict clusters for all test points at once

    if svm is None:
        test_clusters = kMeans.predict(df_tp[['lat', 'lng']].values)
    else:
        test_clusters = svm.predict(df_tp[['lat', 'lng']])

    # Organize test points by cluster
    df_tp['cluster'] = test_clusters
    cluster_groups = df_tp.groupby('cluster')

    locations = []
    errors = []
    total_rps = 0
    for cluster, group in cluster_groups:
        rps = df_rp[df_rp['cluster'] == cluster]
        total_rps += len(group) * len(rps)
        TP_est_location, k_avg_error = process_test_points(group, rps, unique_npcis, rf_param, k_max)
        errors.append(k_avg_error)

    total_rps = int(total_rps / len(df_tp))
    return None, np.vstack(errors), total_rps


def process_test_points(df_tp_cluster, df_rp, unique_npcis, rf_param, k_max):
    # Get reference points in the current cluster

    # Precompute matrices for all reference points in the cluster
    m_rfp, idx_rfp = create_point_matrix(df_rp, unique_npcis, rf_param)

    m_tp, idx_tp = create_point_matrix(df_tp_cluster, unique_npcis, rf_param)

    W, idx_sort = compute_weights(m_rfp, idx_rfp, m_tp, idx_tp)

    TP_est_location, k_avg_error = wknn(df_tp_cluster, df_rp, idx_sort, W, k_max)

    return TP_est_location, k_avg_error
