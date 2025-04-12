import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from scripts.data_filter import filter_dataframe
from scripts.data_loader import load_dataframe
from scripts.data_writer import save_experiment_result
from scripts.utils import (
    NETWORK_TYPE,
    RF_PARAM_5G,
    extract_unique_npcis,
)
from scripts.weighted_coverage import run_weighted_coverage


def load_data(
    selected_campaigns: list[int],
    operator_choice: list[int],
):
    filename = "5G_data_2023.mat"

    # Series of random seeds for reproducability
    random_seeds = np.loadtxt("data/random_seeds.csv", dtype=int)

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
            "rsrq",
            "rssi",
            "rsrp",
            "sinr",
        ],
        campaigns=selected_campaigns,
    )

    return df, random_seeds


def single_run(
    i, filtered_df, rf_param, unique_npcis, random_seed, n_clusters, k_wknn, n_runs
):
    print(f"🔄 Running ({i + 1}/{n_runs} runs) on PID: {os.getpid()}")
    return run_weighted_coverage(
        df=filtered_df,
        rf_param=rf_param,
        cluster_rf_param=rf_param,
        k_max=k_wknn,
        unique_npcis=unique_npcis,
        random_seed=random_seed,
        n_clusters=n_clusters,
    )


def run_experiment(
    df: pd.DataFrame,
    random_seeds: np.ndarray,
    n_runs: int,
    max_k_wknn: int,
    rf_param: RF_PARAM_5G,
    clustering_rf_param: RF_PARAM_5G,
    n_clusters: int,
    operator_choice: list[int],
):

    k_range = range(1, max_k_wknn + 1)

    errors_dict = {k: [] for k in k_range}  # Store the errors
    unique_npcis = extract_unique_npcis(df["measurements_matrix"])

    print(
        f"""
    Running frequency experiment
    🧪 Experiment setup 🧪
    🔢 max k-value for wKNN = {max_k_wknn}
    👨‍👩‍👦‍👦 n clusters = {n_clusters}
    🛜 RF PARAM {rf_param.value}
    📡 Cluster RF PARAM {clustering_rf_param.value}
    📶 Operator choice {operator_choice}
    🔁 Number of runs {n_runs}|
    _________________________________
    """
    )
    results = []
    # Use ProcessPoolExecutor to parallelize the runs
    for k in k_range:
        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = [
                (
                    executor.submit(
                        single_run,
                        i,
                        df,
                        rf_param,
                        unique_npcis,
                        random_seeds[i],
                        0,
                        k,
                        n_runs,
                    ),
                    i,
                )
                for i in range(n_runs)
            ]
            for future, run in futures:
                res, runtime, num_tps = future.result()

                data = np.array([rf_param.value, k, runtime, run])
                data_2d = np.tile(data, (res.shape[0], 1))

                res = np.concatenate([data_2d, res], axis=1)

                results.extend(res)

        print(f"\r✅ k={k} completed                                           ")

    return pd.DataFrame(
        results,
        columns=["rf_param", "wknn_k", "runtime", "run", "errors", "complexity"],
    )


def main():
    # Parameters
    n_runs = 20
    k_wknn = 10
    rf_param = RF_PARAM_5G.SINR
    operator_choice = [1, 10, 50, 88]
    selected_campaigns = list(range(1, 21))

    params = [RF_PARAM_5G.SINR, RF_PARAM_5G.RSRQ, RF_PARAM_5G.RSRP, RF_PARAM_5G.RSSI]

    df, random_seeds = load_data(selected_campaigns, operator_choice)

    start_time = time.time()

    results_df = pd.DataFrame()
    for p in params:
        res_df = run_experiment(
            df,
            random_seeds,
            n_runs,
            k_wknn,
            p,
            p,
            0,
            operator_choice,
        )
        results_df = pd.concat([results_df, res_df], ignore_index=True, axis=0)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total runtime was {total_time} seconds")

    config = {
        "max_wknn_k": k_wknn,
        "rf_param": rf_param.value,
        "operator_choice": operator_choice,
        "n_runs": n_runs,
        "campaigns": selected_campaigns,
        "runtime": total_time,
    }

    data = {
        "data": results_df,
    }

    save_experiment_result("wknn_experiment", config, data)


if __name__ == "__main__":
    main()
