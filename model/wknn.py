import numpy as np
import pandas as pd

from scripts.matrix_operations import create_point_matrix, compute_weights
from scripts.utils import RF_PARAM_5G
from scripts.weighted_coverage import wknn


def predict_lat_lng_wknn(
    df_tp: pd.DataFrame,
    df_rp: pd.DataFrame,
    unique_npcis: np.array,
    rf_param: RF_PARAM_5G,
    wknn_k: int,
    rf_model,
):
    """
    Process points using wKNN with clustering.
    TPs are grouped by predicted cluster membership, then
    WkNN is run with RS with same cluster membership as TPs.

    :param tps: The test points.
    :param rps: The reference points.
    :param unique_npcis: Set of unique pcis.
    :param rf_param: RF parameter for weighted coverage.
    :param wknn_k: K value for WkNN
    :return: estimated positions. np array of (lat, lng) in original order.
    """

    # Predict clusters for all test points
    tp_features, _ = create_point_matrix(df_tp, unique_npcis, rf_param)
    test_clusters = rf_model.predict(tp_features)
    df_tp = df_tp.copy()
    df_tp["predicted_cluster"] = test_clusters

    # Prepare result array
    predicted_positions = np.full((df_tp.shape[0], 2), np.nan)

    # Process each cluster
    for cluster in np.unique(test_clusters):
        idx = np.where(test_clusters == cluster)[0]
        tp_cluster = df_tp.iloc[idx]
        rp_cluster = df_rp[df_rp["cluster"] == cluster]
        if rp_cluster.empty:
            continue

        est_locs = process_points(
            tp_cluster, rp_cluster, unique_npcis, rf_param, wknn_k
        )

        # Assign estimated positions in the correct order
        predicted_positions[idx, :] = est_locs

    return predicted_positions


def predict_lat_lng_wknn_no_cluster(
    df_tp: pd.DataFrame,
    df_rp: pd.DataFrame,
    unique_npcis: np.array,
    rf_param: RF_PARAM_5G,
    wknn_k: int,
):
    """
    Process points using wKNN without clustering.

    :param tps: The test points.
    :param rps: The reference points.
    :param unique_npcis: Set of unique pcis.
    :param rf_param: RF parameter for weighted coverage.
    :param wknn_k: K value for WkNN
    :return: estimated positions. np array of (lat, lng) in original order.
    """
    est_locs = process_points(df_tp, df_rp, unique_npcis, rf_param, wknn_k)
    return est_locs


def process_points(
    tps: pd.DataFrame,
    rps: pd.DataFrame,
    unique_npcis: np.array,
    rf_param: RF_PARAM_5G,
    wknn_k: int,
):
    """
    Process points using wKNN.

    :param tps: The test points.
    :param rps: The reference points.
    :param unique_npcis: Set of unique pcis.
    :param rf_param: RF parameter for weighted coverage.
    :param wknn_k: K value for WkNN
    :return: estimated positions. np array of (lat, lng) in original order.
    """
    # Create point matrices
    m_rp_full, idx_rp_full = create_point_matrix(rps, unique_npcis, rf_param)
    m_tp_full, idx_tp_full = create_point_matrix(tps, unique_npcis, rf_param)

    # Compute weights and sorted indices
    W, idx_sort = compute_weights(m_rp_full, idx_rp_full, m_tp_full, idx_tp_full)

    # Run wKNN
    est_locs, _ = wknn(tps, rps, idx_sort, W, k=wknn_k)
    return est_locs
