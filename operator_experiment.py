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
    use_beam_matching,
):
    print(f"🔄 Running for operator {op} ({i + 1}/{n_runs} runs) on PID: {os.getpid()}")
    _, errors, _, _ = run_weighted_coverage(
        df=filtered_df,
        rf_param=rf_param,
        cluster_rf_param=rf_param,
        k_max=k_wknn,
        unique_npcis=unique_npcis,
        random_seed=random_seed,
        n_clusters=n_clusters,
        use_beam_matching=use_beam_matching,
    )
    return errors.mean()


def run_experiment(
    df: pd.DataFrame,
    random_seeds: np.ndarray,
    n_runs: int,
    k_wknn: int,
    rf_param: RF_PARAM_5G,
    clustering_rf_param: RF_PARAM_5G,
    n_clusters: int,
    operator_choice: list[int],
    use_best_beams: bool = False,
):
    print(
        f"""
    Running frequency experiment
    🧪 Experiment setup 🧪
    🔢 k-value for wKNN = {k_wknn}
    👨‍👩‍👦‍👦 n clusters = {n_clusters}
    🛜 RF PARAM {rf_param.value}
    📡 Cluster RF PARAM {clustering_rf_param.value}
    📶 Operator choice {operator_choice}
    🔁 Number of runs {n_runs}|
    _________________________________
    """
    )

    errors_dict = {op: [] for op in operator_choice}  # Store the errors
    num_entries_dict = {op: [] for op in operator_choice}  # Store the errors
    num_rps_dict = {op: [] for op in operator_choice}

    for operator in operator_choice:

        filtered_df = filter_dataframe(df=df.copy(), operators=[operator])

        unique_npcis = extract_unique_npcis(filtered_df["measurements_matrix"])

        num_entries_dict[operator] = [
            filtered_df["measurements_matrix"].apply(len).sum()
        ]
        num_rps_dict[operator] = [len(filtered_df)]

        # Use ProcessPoolExecutor to parallelize the runs
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [
                executor.submit(
                    single_run,
                    operator,
                    i,
                    filtered_df,
                    rf_param,
                    unique_npcis,
                    random_seeds[i],
                    n_clusters,
                    k_wknn,
                    n_runs,
                    use_best_beams,
                )
                for i in range(n_runs)
            ]
            for future in futures:
                errors_dict[operator].append(future.result())

        print(f"\r✅ {operator} completed                                           ")

    errors_df = pd.DataFrame(errors_dict)
    entries_df = pd.DataFrame(num_entries_dict)
    rps_dict = pd.DataFrame(num_rps_dict)

    return errors_df, entries_df, rps_dict


def main():
    # Parameters
    n_runs = 10
    k_wknn = 2
    rf_param = RF_PARAM_5G.RSRQ
    clustering_rf_param = RF_PARAM_5G.RSRQ
    n_clusters = 5
    operator_choice = [1, 10, 50, 88]
    selected_campaigns = None
    use_best_beams = False

    start_time = time.time()

    # vodafone
    df, random_seeds = load_data(selected_campaigns, rf_param, operator_choice)

    errors_df, entries_df, rps_df = run_experiment(
        df,
        random_seeds,
        n_runs,
        k_wknn,
        rf_param,
        clustering_rf_param,
        n_clusters,
        operator_choice,
    )

    # control

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
        "use_best_beams": use_best_beams,
    }

    data = {
        "errors": errors_df,
        "entries": entries_df,
        "rps": rps_df,
    }

    save_experiment_result("operator_experiment", config, data)


if __name__ == "__main__":
    main()
