import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from scripts.utils import (
    RF_PARAM,
    get_miss_ref_value,
    get_unique_npcis,
    dataset_reference_test_split,
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
    num_unique_npcis = len(unique_npcis)
    miss_ref_value = get_miss_ref_value(rf_param)

    point_matrix = np.full(
        shape=[num_points, num_unique_npcis], fill_value=miss_ref_value
    )

    idx_matrix = np.zeros(shape=[num_points, num_unique_npcis])

    for i in range(num_points):
        measurements = df.iloc[i]["measurements_matrix"]
        for _, row in measurements.iterrows():
            npc = row["NPCI"]
            rf_value = row[rf_param.value]
            if npc in unique_npcis:
                idx = np.where(unique_npcis == npc)[0][0]
                point_matrix[i, idx] = (
                    rf_value if not np.isnan(rf_value) else miss_ref_value
                )
                idx_matrix[i, idx] = 1 if not np.isnan(rf_value) else 0

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
    :return: Weights and sorted indecies by weight
    """
    # Caclulate distances between tps and rps
    D = cdist(m_tp, m_rfp, metric="euclidean")

    # Normalize distances based on common NPCIs
    for i in range(m_tp.shape[0]):
        match = np.logical_and(idx_tp[i, :], idx_rfp)
        s = np.sum(match, axis=1)
        z = np.where(s == 0)
        nz = np.where(s != 0)

        for j in nz[0]:
            D[i, j] = D[i, j] / s[j]
        for j in z[0]:
            D[i, j] = np.inf  # Use np.inf to represent a very large distance

    # Set distances to dummy reference points to a very large value
    dummy_rfps = np.all(idx_rfp == 0, axis=1)
    D[:, dummy_rfps] = np.inf

    # Replace zero distances with a small value to avoid singularities
    D[D == 0] = np.min(D[D != 0]) / 20

    # Sort distances and compute weights
    D_sort = np.sort(D, axis=1)
    idx_sort = np.argsort(D, axis=1)
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
        lat_k_TP = sum_lat / np.sum(W[:, :this_k], axis=1)
        long_k_TP = sum_long / np.sum(W[:, :this_k], axis=1)

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


def run_weighted_coverage(
        dataset: pd.DataFrame, rf_param: RF_PARAM, k_max: int
) -> (float, float):
    """
    'Main' entry point.
    Splits the dataset into test and reference points.
    Creates the point matrecies and calculates the weights.
    Runs wKNN to estimate position and calculate error.
    :param dataset: Original dataset
    :param rf_param: What rf param to use
    :param k_max: Max number of neighbors for wKNN
    :return: Estimated locations and average error for each k value
    """
    # Set the probability for a point to be a test point
    TP_probability = 0.3

    # Randomly assign points as test points (2) or reference points (1)
    df_tp, df_rp = dataset_reference_test_split(dataset, TP_probability)

    # Get unique NPCIs
    unique_npcis = get_unique_npcis(dataset)

    # Create matrices for test and reference points
    m_rfp, idx_rfp = create_point_matrix(df_rp, unique_npcis, rf_param)
    m_tp, idx_tp = create_point_matrix(df_tp, unique_npcis, rf_param)

    # Compute weights for wKNN
    W, idx_sort = compute_weights(m_rfp, idx_rfp, m_tp, idx_tp)

    tp_est_location, k_avg_error = wknn(df_tp, df_rp, idx_sort, W, k_max)

    return tp_est_location, k_avg_error
