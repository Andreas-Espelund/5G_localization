import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from scripts.clustering import train_kmeans, train_random_forest
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
    idx_sort: np.ndarray[int],
    W: np.ndarray[np.float64],
    k: int,
) -> (np.array, float):
    """
    The WkNN algorithm

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
    use_pca: bool = False,
) -> (np.array, np.array, int, float):
    """
    Experimental method for running weighted coverage algorithm in experiments.

    :param df: Dataframe of reference points
    :param rf_param: RF parameter for WkNN
    :param cluster_rf_param: Param for clustering?
    :param k_max: K value for wKNN
    :param unique_npcis: Set of unique npcis (features)
    :param random_seed: Random seed for reproducibility
    :param n_clusters: Number of clusters for kmeans
    :param use_pca: Should use PCA?
    :return: Runtime, complexity and error measurements
    """

    tmp = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    if n_clusters > 0:
        train_kmeans(tmp, n_clusters, random_seed)

    df_tp, df_rp = dataset_tp_rp_split(tmp, 0.3, random_seed)

    if not n_clusters > 0:
        start_time = time.perf_counter()
        results = process_test_points(
            df_tp, df_rp, unique_npcis, rf_param, k_max, use_pca=use_pca
        )
        end_time = time.perf_counter()

        runtime = end_time - start_time

        return results, runtime, df_tp.shape[0]

    rf_model = train_random_forest(df_rp, unique_npcis, rf_param, 100, random_seed)

    start_time = (
        time.perf_counter()
    )  # don't include model training in the online stage timing
    result = process_clusters(
        df_tp,
        df_rp,
        unique_npcis,
        rf_param,
        cluster_rf_param,
        k_max,
        rf_model,
        use_pca=use_pca,
    )
    end_time = time.perf_counter()

    runtime = end_time - start_time

    return result, runtime, df_tp.shape[0]


def process_clusters(
    df_tp: pd.DataFrame,
    df_rp: pd.DataFrame,
    unique_npcis: np.array(tuple[int, int, int]),
    rf_param: RF_PARAM_5G,
    cluster_rf_param: RF_PARAM_5G,
    k_max: int,
    rf_model,
    use_pca: bool,
):
    """
    Predict cluster membership and process points grouped on
    cluster.

    :param rf_model: Random forest model
    :param cluster_rf_param: Param to use for cluster prediction in Random Forest
    :param df_tp: Test points dataframe
    :param df_rp: Reference points dataframe
    :param pcis: Set of unique npcis (features)
    :param rf_param: RF parameter for WkNN
    :param k: K value for wKNN
    :param use_pca: Should use PCA?
    :param n_components: Number of components for PCA
    :return:
    """
    # Predict clusters for all test points at once
    tp_features, _ = create_point_matrix(df_tp, unique_npcis, cluster_rf_param)

    # Organize test points by cluster
    test_clusters = rf_model.predict(tp_features)
    df_tp["predicted_cluster"] = test_clusters

    cluster_groups = df_tp.groupby("predicted_cluster")

    # Initialize lists to store results
    results = []

    for cluster, group in cluster_groups:
        rps = df_rp[df_rp["cluster"] == cluster]

        # Process each cluster's test points
        res = process_test_points(
            group, rps, unique_npcis, rf_param, k_max, use_pca=use_pca
        )
        results.append(res)

    # Concatenate all estimated locations and average errors

    return np.vstack(results)


def process_test_points(
    df_tp: pd.DataFrame,
    df_rp: pd.DataFrame,
    pcis: list[tuple],
    rf_param: RF_PARAM_5G,
    k: int = 2,
    use_pca: bool = False,
    n_components: float = 0.95,
):
    """
    Process a batch of test points.
    :param df_tp: Test points dataframe
    :param df_rp: Reference points dataframe
    :param pcis: Set of unique npcis (features)
    :param rf_param: RF parameter for WkNN
    :param k: K value for wKNN
    :param use_pca: Should use PCA?
    :param n_components: Number of components for PCA
    :return: Results
    """

    n_points = df_tp.shape[0]

    # Create the full point matrix with all beam features
    m_rp_full, idx_rp_full = create_point_matrix(df_rp, pcis, rf_param)
    m_tp_full, idx_tp_full = create_point_matrix(df_tp, pcis, rf_param)

    # If we dont use PCA, just calcualte the errors and return
    if not use_pca:
        W, idx_sort = compute_weights(m_rp_full, idx_rp_full, m_tp_full, idx_tp_full)
        _, errors = wknn(df_tp, df_rp, idx_sort, W, k=k)
        complexity = np.repeat(m_rp_full.shape[0] * m_tp_full.shape[1], n_points)

        return np.array([errors, complexity]).T

    # Apply PCA to reduce dimensions
    pca = PCA(n_components=n_components)
    pca.fit(m_rp_full)
    m_rp_pca = pca.transform(m_rp_full)
    m_tp_pca = pca.transform(m_tp_full)

    W_pca, idx_sort_pca = compute_weights_pca(m_rp_pca, m_tp_pca)

    _, errors = wknn(df_tp, df_rp, idx_sort_pca, W_pca, k)

    complexity = np.repeat(m_rp_pca.shape[0] * m_tp_pca.shape[1], n_points)

    return np.array([errors, complexity]).T
