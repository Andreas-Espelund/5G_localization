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


def single_run(
    op,
    i,
    filtered_df,
    rf_param,
    unique_npcis,
    random_seed,
    n_clusters,
    k_wknn,
    n_runs,
    n_best_beams,
):
    print(
        f"🔄 Running for operator {op} - {n_best_beams} ({i + 1}/{n_runs} runs) on PID: {os.getpid()}"
    )
    return run_weighted_coverage(
        df=filtered_df.copy(),
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
    k_wknn: int,
    rf_param: RF_PARAM_5G,
    clustering_rf_param: RF_PARAM_5G,
    cluster_range: int,
    operator_choice: list[int],
    use_pca,
):
    print(
        f"""
    Running frequency experiment
    🧪 Experiment setup 🧪
    🔢 k-value for wKNN = {k_wknn}
    👨‍👩‍👦‍👦 n clusters = {cluster_range}
    🛜 RF PARAM {rf_param.value}
    📡 Cluster RF PARAM {clustering_rf_param.value}
    📶 Operator choice {operator_choice}
    🔁 Number of runs {n_runs}|
    _________________________________
    """
    )

    highest_frequencies = {
        1: [648768],
        10: [643296, 643295],
        50: [641663, 641664],
        88: [638015, 638016],
    }

    results = []

    for operator in operator_choice:
        nr_arfcns = highest_frequencies[operator]

        filtered_df = filter_dataframe(
            df=df.copy(), operators=[operator], freqs=nr_arfcns
        )
        for n_clus in cluster_range:

            unique_npcis = extract_unique_npcis(filtered_df["measurements_matrix"])

            # Use ProcessPoolExecutor to parallelize the runs
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [
                    executor.submit(
                        run_weighted_coverage,
                        filtered_df,
                        rf_param,
                        rf_param,
                        k_wknn,
                        unique_npcis,
                        random_seeds[i],
                        n_clus,
                        use_pca,
                    )
                    for i in range(n_runs)
                ]
                for future in futures:
                    data, runtime = future.result()

                    op = np.array([operator, n_clus, use_pca, runtime])
                    op_2d = np.tile(op, (data.shape[0], 1))

                    res = np.concatenate([op_2d, data], axis=1)

                    results.extend(res)
                    print(f"Operator {operator} run done in {runtime}")

        print(f"✅ Operator {operator} completed ✅")

    results_df = pd.DataFrame(
        results,
        columns=["operator", "n_clusters", "use_pca", "runtime", "error", "complexity"],
    )

    return results_df


def main():
    # Parameters
    n_runs = 20
    k_wknn = 2
    rf_param = RF_PARAM_5G.RSRQ
    clustering_rf_param = RF_PARAM_5G.RSRQ
    cluster_range = range(10, 11)
    operator_choice = [1, 10, 50, 88]
    selected_campaigns = list(range(1, 31))

    start_time = time.time()

    # vodafone
    df, random_seeds = load_data(selected_campaigns, rf_param, operator_choice)

    result_df = run_experiment(
        df,
        random_seeds,
        n_runs,
        k_wknn,
        rf_param,
        clustering_rf_param,
        cluster_range,
        operator_choice,
        use_pca=True,
    )

    result_df_control = run_experiment(
        df,
        random_seeds,
        n_runs,
        k_wknn,
        rf_param,
        clustering_rf_param,
        cluster_range,
        operator_choice,
        use_pca=False,
    )

    result_df = pd.concat([result_df, result_df_control], ignore_index=True)

    # control

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total runtime was {total_time} seconds")

    config = {
        "wknn_k": k_wknn,
        "rf_param": rf_param.value,
        "cluster_rf_param": clustering_rf_param.value,
        "operator_choice": operator_choice,
        "n_clusters": list(cluster_range),
        "n_runs": n_runs,
        "campaigns": "all",
        "runtime": total_time,
    }

    data = {
        "results": result_df,
    }

    save_experiment_result("operator_experiment", config, data)


if __name__ == "__main__":
    main()
