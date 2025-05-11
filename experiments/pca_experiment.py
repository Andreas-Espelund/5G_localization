import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from scripts.data_filter import filter_dataframe
from scripts.data_loader import load_dataframe
from scripts.matrix_operations import create_point_matrix, compute_weights
from scripts.utils import (
    NETWORK_TYPE,
    RF_PARAM_5G,
)
from scripts.weighted_coverage import wknn_one


def load_data(
    selected_campaigns: list[int], rf_param: RF_PARAM_5G, operator_choice: list[int]
):

    filename = "5G_data_2023.mat"

    # Series of random seeds for reproducability
    random_seeds = np.loadtxt("../data/random_seeds.csv", dtype=int)

    # load the dataframe from saved file or 'raw' matlab file
    df = load_dataframe(filename, NETWORK_TYPE._5G)

    # # Drop unused columns to save space
    matrix_cols_to_drop = ["toa_pps", "toa_cir", "toa_cov", "campaign_id"]
    df["measurements_matrix"] = df["measurements_matrix"].apply(
        lambda x: x.drop(columns=matrix_cols_to_drop)
    )

    # Data filtering
    df = filter_dataframe(
        df=df,
        operators=operator_choice,
        include_columns=[
            "pci",
            "beam_index",
            "nr_arfcn",
            "operator_id",
            rf_param.value,
        ],
        campaigns=selected_campaigns,
    )
    return df, random_seeds


def compute_weights_pca(m_rfp_pca, m_tp_pca):
    """
    Compute weights using PCA-transformed data
    """
    # Compute Euclidean distances in the PCA space
    D = cdist(m_tp_pca, m_rfp_pca, metric="euclidean")

    # Sort distances and compute weights
    idx_sort = np.argsort(D, axis=1)
    D_sort = np.take_along_axis(D, idx_sort, axis=1)

    # Avoid division by zero
    min_nonzero_distance = np.min(D[D > 0]) if np.any(D > 0) else 0.1
    D_sort[D_sort == 0] = min_nonzero_distance / 20

    W = 1.0 / D_sort
    return W, idx_sort


def pca_strategy(
    df_tp: pd.DataFrame,
    df_rp: pd.DataFrame,
    pcis: list[tuple],
    rf_param: RF_PARAM_5G,
    k: int = 2,
):
    # 1. Create the full point matrix with all beam features
    m_rp_full, idx_rp_full = create_point_matrix(df_rp, pcis, rf_param)
    m_tp_full, idx_tp_full = create_point_matrix(df_tp, pcis, rf_param)

    # 3. Apply PCA to reduce dimensions
    pca = PCA(n_components=0.95)
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


def main():
    n_runs = 10
    k_wknn = 2
    rf_param = RF_PARAM_5G.RSRQ
    operator_choice = [10]
    selected_campaigns = list(range(1, 21))

    df, random_seeds = load_data(selected_campaigns, rf_param, operator_choice)

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.submit(pca_strategy, df, df, rf_param, k_wknn)
