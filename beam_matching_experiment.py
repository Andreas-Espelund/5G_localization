import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from scripts.beamforming import get_best_beam
from scripts.data_filter import filter_dataframe
from scripts.data_loader import load_dataframe
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
    i,
    filtered_df,
    rf_param,
    unique_npcis,
    random_seed,
    k_wknn,
    n_runs,
    n_clusters,
    use_beam_matching,
):
    print(f"🔄 Running  ({i + 1}/{n_runs} runs) on PID: {os.getpid()}")

    _, errors, complexity, runtime = run_weighted_coverage(
        df=filtered_df,
        rf_param=rf_param,
        cluster_rf_param=rf_param,
        unique_npcis=unique_npcis,
        random_seed=random_seed,
        k_max=k_wknn,
        n_clusters=n_clusters,
        use_beam_matching=use_beam_matching,
    )
    return errors.mean(), complexity, runtime


def run_experiment(
    df: pd.DataFrame,
    random_seeds: np.ndarray,
    n_runs: int,
    k_wknn: int,
    rf_param: RF_PARAM_5G,
    n_clusters: int,
    use_beam_matching: bool,
):

    data = []

    unique_npcis = extract_unique_npcis(df["measurements_matrix"])

    # find the best beams for each tp for later matching between TP and RP
    df["best_beam"] = df["measurements_matrix"].apply(
        lambda x: get_best_beam(x, rf_param)
    )

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
                k_wknn,
                n_runs,
                n_clusters,
                use_beam_matching,
            )
            for i in range(n_runs)
        ]
        for future in futures:
            errors, complexity, runtime = future.result()
            data.append((errors, complexity, runtime))

    data_df = pd.DataFrame(data, columns=["errors", "complexity", "runtime"])

    return data_df


def main():
    # Parameters
    n_runs = 30
    k_wknn = 2
    rf_param = RF_PARAM_5G.RSRQ
    clustering_rf_param = RF_PARAM_5G.RSRQ
    operator_choice = [10]
    selected_campaigns = list(range(1, 41))
    n_clusters = 0
    df, random_seeds = load_data(selected_campaigns, rf_param, operator_choice)

    start_time = time.time()

    data_df = run_experiment(
        df,
        random_seeds,
        n_runs,
        k_wknn,
        rf_param,
        n_clusters,
        True,
    )

    df, random_seeds = load_data(selected_campaigns, rf_param, operator_choice)

    # run without beam matching as controle
    control_data_df = run_experiment(
        df,
        random_seeds,
        n_runs,
        k_wknn,
        rf_param,
        n_clusters,
        False,
    )

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total runtime was {total_time} seconds")
    print(f"errors", data_df)
    print(f"control errors", control_data_df)

    config = {
        "wknn_k": k_wknn,
        "rf_param": rf_param.value,
        "cluster_rf_param": clustering_rf_param.value,
        "operator_choice": operator_choice,
        "n_clusters": n_clusters,
        "n_runs": n_runs,
        "campaigns": selected_campaigns,
        "runtime": total_time,
    }

    data = {
        "data": data_df,
        "control": control_data_df,
    }

    # save_experiment_result("beam_matching_experiment", config, data)


if __name__ == "__main__":
    main()
