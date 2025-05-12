import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from scripts.clustering import train_random_forest
from scripts.data_filter import filter_dataframe
from scripts.data_loader import load_dataframe
from scripts.data_writer import save_experiment_result
from scripts.utils import (
    NETWORK_TYPE,
    RF_PARAM_5G,
    extract_unique_npcis,
    dataset_tp_rp_split,
)
from scripts.weighted_coverage import process_clusters


def load_data(
    selected_campaigns: list[int], rf_param: RF_PARAM_5G, operator_choice: list[int]
):
    filename = "5G_data_2023.mat"

    # Series of random seeds for reproducability
    random_seeds = np.loadtxt("../config/random_seeds.csv", dtype=int)

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


def run_experiment(
    df: pd.DataFrame,
    random_seeds: np.ndarray,
    k_wknn: int,
    rf_param: RF_PARAM_5G,
    n_clusters: int,
):

    random_seed = random_seeds[n_clusters]

    pcis = extract_unique_npcis(df["measurements_matrix"])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)

    features = df[["lat", "lng"]].values

    df["cluster"] = kmeans.fit_predict(features)

    df_tp, df_rp = dataset_tp_rp_split(df, 0.3, random_seed)
    rf_model = train_random_forest(
        df=df_rp,
        unique_npcis=pcis,
        rf_param=rf_param,
        random_seed=random_seed * 42,
        n_estimators=100,
    )

    _ = process_clusters(
        df_tp,
        df_rp,
        pcis,
        rf_param,
        rf_param,
        k_wknn,
        rf_model,
        use_pca=False,
    )

    return df_tp


def main():
    # Parameters
    n_runs = 1
    k_wknn = 2
    rf_param = RF_PARAM_5G.RSRQ
    clustering_rf_param = RF_PARAM_5G.RSRQ
    n_clusters = 10
    operator_choice = [1, 10, 50, 88]
    selected_campaigns = list(range(1, 31))

    start_time = time.time()

    # vodafone
    df, random_seeds = load_data(selected_campaigns, rf_param, operator_choice)

    df_tp = run_experiment(
        df=df,
        random_seeds=random_seeds,
        k_wknn=2,
        rf_param=rf_param,
        n_clusters=n_clusters,
    )

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total runtime was {total_time} seconds")

    config = {
        "wknn_k": k_wknn,
        "rf_param": rf_param.value,
        "cluster_rf_param": clustering_rf_param.value,
        "operator_choice": operator_choice,
        "n_clusters": n_clusters,
        "n_runs": n_runs,
        "campaigns": "all",
        "runtime": total_time,
    }
    print(df_tp.head(2))

    data = {
        "results": df_tp,
    }

    save_experiment_result("misclassification_experiment", config, data)


if __name__ == "__main__":
    main()
