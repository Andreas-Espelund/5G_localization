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


def load_data(selected_campaigns: list[int]):
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
        operators=[10],
        include_columns=["pci", "beam_index", "nr_arfcn", "operator_id", "rsrq"],
        campaigns=selected_campaigns,
    )

    return df, random_seeds


def single_run(
    i, filtered_df, rf_params, unique_npcis, random_seed, n_clusters, k_wknn, n_runs
):
    print(
        f"🔄 Running for nr_arfcn {n_clusters} ({i + 1}/{n_runs} runs) on PID: {os.getpid()}"
    )
    _, errors, complexity, runtime = run_weighted_coverage(
        df=filtered_df,
        rf_params=rf_params,
        cluster_rf_params=rf_params,
        k_max=k_wknn,
        unique_npcis=unique_npcis,
        random_seed=random_seed,
        n_clusters=n_clusters,
    )
    return errors.mean(), complexity, runtime


def run_experiment(
    df: pd.DataFrame,
    random_seeds: np.ndarray,
    n_runs: int,
    k_wknn: int,
    rf_params: list,
    clustering_rf_params: list,
    cluster_range: range,
    operator_choice: list[int],
):

    print(
        f"""
    Running frequency experiment
    🧪 Experiment setup 🧪
    🔢 k-value for wKNN = {k_wknn}
    👨‍👩‍👦‍👦 cluster range= {cluster_range}
    🛜 RF PARAM {str(rf_params)}
    📡 Cluster RF PARAM {str(clustering_rf_params)}
    📶 Operator choice {operator_choice}
    🔁 Number of runs {n_runs}|
    _________________________________
    """
    )

    errors_dict = {c: [] for c in cluster_range}  # Store the errors
    complexity_dict = {c: [] for c in cluster_range}
    runtime_dict = {c: [] for c in cluster_range}

    unique_npcis = extract_unique_npcis(df["measurements_matrix"])

    for n_clusters in cluster_range:

        # Use ProcessPoolExecutor to parallelize the runs
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [
                executor.submit(
                    single_run,
                    i,
                    df,
                    rf_params,
                    unique_npcis,
                    random_seeds[i],
                    n_clusters,
                    k_wknn,
                    n_runs,
                )
                for i in range(n_runs)
            ]
            for future in futures:
                errors, complexity, runtime = future.result()
                errors_dict[n_clusters].append(errors)
                complexity_dict[n_clusters].append(complexity)
                runtime_dict[n_clusters].append(runtime)

        print(f"\r✅ {n_clusters} completed                                           ")

    errors_df = pd.DataFrame(errors_dict)
    complexity_df = pd.DataFrame(complexity_dict)
    runtime_df = pd.DataFrame(runtime_dict)

    return errors_df, complexity_df, runtime_df


def main():
    # Parameters
    n_runs = 50
    k_wknn = 2
    rf_params = [RF_PARAM_5G.RSRQ]
    clustering_rf_params = [RF_PARAM_5G.RSRQ]
    cluster_range = range(0, 11)
    operator_choice = [10]
    selected_campaigns = list(range(1, 21))

    df, random_seeds = load_data(selected_campaigns)

    start_time = time.time()

    errors_df, complexity_df, runtime_df = run_experiment(
        df,
        random_seeds,
        n_runs,
        k_wknn,
        rf_params,
        clustering_rf_params,
        cluster_range,
        operator_choice,
    )

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total runtime was {total_time} seconds")

    config = {
        "wknn_k": k_wknn,
        "rf_param": rf_params[0].value,
        "cluster_rf_param": clustering_rf_params[0].value,
        "operator_choice": operator_choice,
        "cluster_range": list(cluster_range),
        "n_runs": n_runs,
        "campaigns": selected_campaigns,
        "runtime": total_time,
    }

    data = {
        "errors": errors_df,
        "complexity": complexity_df,
        "runtime": runtime_df,
    }

    save_experiment_result("clustering_experiment", config, data)


if __name__ == "__main__":
    main()
