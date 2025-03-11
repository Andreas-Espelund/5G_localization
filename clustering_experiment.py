import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from scripts.beamforming import filter_best_beams
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
    selected_campaigns: list[int], rf_param: RF_PARAM_5G, use_beam_filter: bool = False
) -> pd.DataFrame:
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
        campaigns=selected_campaigns,
    )
    if use_beam_filter:
        # Beam filtering. only include the best beam for each pci
        df["measurements_matrix"] = df["measurements_matrix"].apply(
            lambda x: filter_best_beams(x, rf_param)
        )

    return df, random_seeds


def single_run(
    i, filtered_df, rf_param, unique_npcis, random_seed, n_clusters, k_wknn, n_runs
):
    print(
        f"🔄 Running for cluster {n_clusters} ({i + 1}/{n_runs} runs) on PID: {os.getpid()}"
    )
    _, errors, complexity, runtime = run_weighted_coverage(
        df=filtered_df,
        rf_param=rf_param,
        cluster_rf_param=rf_param,
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
    rf_param: RF_PARAM_5G,
    clustering_rf_param: RF_PARAM_5G,
    cluster_range: range,
    operator_choice: list[int],
):

    print(
        f"""
    Running frequency experiment
    🧪 Experiment setup 🧪
    🔢 k-value for wKNN = {k_wknn}
    👨‍👩‍👦‍👦 cluster range= {cluster_range}
    🛜 RF PARAM {rf_param.value}
    📡 Cluster RF PARAM {clustering_rf_param.value}
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
        with ProcessPoolExecutor(max_workers=25) as executor:
            futures = [
                executor.submit(
                    single_run,
                    i,
                    df,
                    rf_param,
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
    n_runs = 5
    k_wknn = 2
    rf_param = RF_PARAM_5G.SINR
    clustering_rf_param = RF_PARAM_5G.SINR
    cluster_range = range(1, 10)
    operator_choice = [10]
    selected_campaigns = list(range(1, 21))

    selected_params = [
        RF_PARAM_5G.DUMMY,
        RF_PARAM_5G.SINR,
        RF_PARAM_5G.RSRQ,
        RF_PARAM_5G.RSRP,
        RF_PARAM_5G.RSSI,
    ]
    start_time = time.time()

    results_error = {}
    results_complexity = {}
    for param in selected_params:
        df, random_seeds = load_data(selected_campaigns, param, False)

        errors_df, complexity_df, runtime_df = run_experiment(
            df,
            random_seeds,
            n_runs,
            k_wknn,
            rf_param if param is RF_PARAM_5G.DUMMY else param,
            param,
            cluster_range,
            operator_choice,
        )

        errors_df["param"] = param.value
        complexity_df["param"] = param.value

        results_error[param] = errors_df
        results_complexity[param] = complexity_df

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total runtime was {total_time} seconds")

    config = {
        "wknn_k": k_wknn,
        "rf_param": rf_param.value,
        "cluster_rf_param": list(map(lambda x: x.value, selected_params)),
        "operator_choice": operator_choice,
        "cluster_range": list(cluster_range),
        "n_runs": n_runs,
        "campaigns": selected_campaigns,
        "runtime": total_time,
    }

    error_result_df = pd.DataFrame()
    for k, v in results_error.items():
        error_result_df = pd.concat([error_result_df, v], ignore_index=True, axis=0)

    complexity_result_df = pd.DataFrame()
    for k, v in results_complexity.items():
        complexity_result_df = pd.concat(
            [complexity_result_df, v], ignore_index=True, axis=0
        )

    data = {
        "errors": error_result_df,
        "complexity": complexity_result_df,
    }

    save_experiment_result("clustering_experiment", config, data)


if __name__ == "__main__":
    main()
