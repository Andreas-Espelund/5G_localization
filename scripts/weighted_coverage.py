import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from scripts.beamforming import find_matching_rps, get_best_beam
from scripts.data_processing import cluster_data_and_train_random_forest
from scripts.matrix_operations import (
    create_point_matrix,
    compute_weights,
    compute_weights_pca,
)
from scripts.utils import (
    RF_PARAM_5G,
    haversine_distance,
    dataset_tp_rp_split,
)


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
    idx_sort: np.ndarray[int],
    W: np.ndarray[np.float64],
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
    rf_param: RF_PARAM_5G,
    cluster_rf_param: RF_PARAM_5G,
    k_max: int,
    unique_npcis: np.array(tuple[int, int, int]),
    random_seed: int,
    n_clusters: int,
    use_beam_matching: bool = False,
) -> (np.array, np.array, int, float):

    if use_beam_matching:
        df["best_beam"] = df["measurements_matrix"].apply(
            lambda x: get_best_beam(x, rf_param)
        )

    tmp = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    df_tp, df_rp = dataset_tp_rp_split(tmp, 0.3, random_seed)

    if not n_clusters > 0:
        start_time = time.time()
        TP_est_location, k_avg_error = process_test_points(
            df_tp, df_rp, unique_npcis, rf_param, k_max, use_beam_matching
        )
        end_time = time.time()
        complexity = len(df_tp) * len(df_rp)
        return TP_est_location, k_avg_error, complexity, end_time - start_time

    rf_model = cluster_data_and_train_random_forest(
        df_rp, n_clusters, unique_npcis, cluster_rf_param, random_seed
    )

    start_time = time.time()  # dont include model training in the online stage timing
    TP_est_location, k_avg_error, rp_factor = process_clusters(
        df_tp,
        df_rp,
        unique_npcis,
        rf_param,
        cluster_rf_param,
        k_max,
        rf_model,
        use_beam_matching,
    )
    end_time = time.time()

    return TP_est_location, k_avg_error, rp_factor, end_time - start_time


def process_clusters(
    df_tp: pd.DataFrame,
    df_rp: pd.DataFrame,
    unique_npcis: np.array(tuple[int, int, int]),
    rf_param: RF_PARAM_5G,
    cluster_rf_param: RF_PARAM_5G,
    k_max: int,
    rf_model,
    use_beam_matching: bool = False,
):
    # Predict clusters for all test points at once
    tp_features, _ = create_point_matrix(df_tp, unique_npcis, cluster_rf_param)

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
            group, rps, unique_npcis, rf_param, k_max, use_beam_matching
        )

        # Store results
        all_tp_est_locations.extend(TP_est_location)
        all_k_avg_errors.extend(k_avg_error)

    # Concatenate all estimated locations and average errors
    TP_est_location_combined = np.vstack(all_tp_est_locations)
    all_k_avg_errors = np.array(all_k_avg_errors)
    return TP_est_location_combined, all_k_avg_errors, total_rps


def process_test_points(
    df_tp: pd.DataFrame,
    df_rp: pd.DataFrame,
    unique_npcis: np.array(tuple[int, int, int]),
    rf_param: RF_PARAM_5G,
    k_max: int,
    use_beam_matching: bool,
):
    # Create the point matrix for the reference points
    m_rfp, idx_rfp = create_point_matrix(df_rp, unique_npcis, rf_param)

    # Create the point matrix for the test points
    m_tp, idx_tp = create_point_matrix(df_tp, unique_npcis, rf_param)

    # Compute the weights between the test points and reference points
    if use_beam_matching:
        df_tp["matches"] = df_tp["best_beam"].apply(
            lambda beam: find_matching_rps(df_rp, beam)
        )
        W, idx_sort = compute_weights(m_rfp, idx_rfp, m_tp, idx_tp)
    else:
        W, idx_sort = compute_weights(m_rfp, idx_rfp, m_tp, idx_tp)

    # Do wKNN to estimate the positions and errors
    TP_est_location, k_avg_error = wknn_one(df_tp, df_rp, idx_sort, W, k_max)

    return TP_est_location, k_avg_error


def process_test_points_pca(
    df_tp: pd.DataFrame,
    df_rp: pd.DataFrame,
    pcis: list[tuple],
    rf_param: RF_PARAM_5G,
    k: int = 2,
    n_components: int = 0.95,
):
    # 1. Create the full point matrix with all beam features
    m_rp_full, idx_rp_full = create_point_matrix(df_rp, pcis, rf_param)
    m_tp_full, idx_tp_full = create_point_matrix(df_tp, pcis, rf_param)

    # 3. Apply PCA to reduce dimensions
    pca = PCA(n_components=n_components)
    pca.fit(m_rp_full)
    m_rp_pca = pca.transform(m_rp_full)
    m_tp_pca = pca.transform(m_tp_full)

    W_pca, idx_sort_pca = compute_weights_pca(m_rp_pca, m_tp_pca)

    W, idx_sort = compute_weights(m_rp_full, idx_rp_full, m_tp_full, idx_tp_full)

    _, errors = wknn_one(df_tp, df_rp, idx_sort_pca, W_pca, k)

    _, errors_control = wknn_one(df_tp, df_rp, idx_sort, W, k=2)

    return (
        errors.mean(),
        int(m_rp_pca.shape[0] * m_rp_pca.shape[1]),
        errors_control.mean(),
        int(m_rp_full.shape[0] * m_rp_full.shape[1]),
    )


def wknn_one_tp_row(
    df_tp: pd.DataFrame,
    df_rp: pd.DataFrame,
    idx_sort: np.ndarray[int],
    W: np.ndarray[np.float64],
    k: int,
) -> (np.array, float):
    """
    :param df_tp: Dataframe of reference points - ASSUMED TO HAVE ONE ROW
    :param df_rp: Dataframe of test points
    :param idx_sort: sorted index matrix by weights
    :param W: the weight matrix for the test/reference points
    :param k: Number of neighbors for wKNN
    :return: Estimated locations and average error for the given k value
    """
    # Extract real positions of the test point (assuming only one row in df_tp)
    real_lat = df_tp["lat"].iloc[0]
    real_long = df_tp["lng"].iloc[0]
    real_position = np.array(
        [[real_lat, real_long]]
    )  # Still needs to be 2D for haversine_distance

    # Select the k-nearest reference points.
    # Since df_tp has one row, we only need the first row of idx_sort and W
    RFP_selected_idx = idx_sort[0, :k]
    W_row = W[0, :k]

    # Extract coordinates of the selected reference points
    lat_k_RFP_matrix = df_rp.iloc[RFP_selected_idx]["lat"].values
    long_k_RFP_matrix = df_rp.iloc[RFP_selected_idx]["lng"].values

    # Compute weighted sums of coordinates
    sum_lat = np.sum(lat_k_RFP_matrix * W_row)
    sum_long = np.sum(long_k_RFP_matrix * W_row)

    # Compute estimated coordinates of test points
    sum_weights = np.sum(W_row)
    lat_k_TP = np.where(sum_weights != 0, sum_lat / sum_weights, np.nan)
    long_k_TP = np.where(sum_weights != 0, sum_long / sum_weights, np.nan)

    # Compute errors using Haversine formula
    errors = haversine_distance(
        real_position[0, 0], real_position[0, 1], lat_k_TP, long_k_TP
    )

    # Store estimated locations
    TP_est_location = np.zeros((1, 2))  # Still 2D array for consistency in return type
    TP_est_location[0, 0] = lat_k_TP
    TP_est_location[0, 1] = long_k_TP

    return (
        TP_est_location,
        errors,
    )
