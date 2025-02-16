import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from scripts.beamforming import get_best_beam, find_matching_rps
from scripts.data_filter import filter_dataframe
from scripts.data_loader import load_dataframe
from scripts.matrix_operations import create_point_matrix
from scripts.single import create_point_vector, compute_weights_single, wknn_single
from scripts.utils import (
    NETWORK_TYPE,
    RF_PARAM_5G,
    extract_unique_npcis,
    dataset_tp_rp_split,
)


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
    use_beam_matching,
):
    print(f"🔄 Running  ({i + 1}/{n_runs} runs) on PID: {os.getpid()}")

    tmp = filtered_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df_tp, df_rp = dataset_tp_rp_split(tmp, 0.3, random_seed)

    # iterate over tps
    errors = []

    df_tp["matches"] = df_tp["best_beam"].apply(lambda x: find_matching_rps(df_rp, x))

    for i, row in df_tp.iterrows():
        if use_beam_matching:
            matches = row["matches"]
            rps = df_rp.loc[matches]
        else:
            rps = df_rp

        test_point = pd.DataFrame([row])
        # Create point matrices/vectors
        m_tp, idx_tp = create_point_vector(test_point, unique_npcis, rf_param)
        m_rp, idx_rp = create_point_matrix(rps, unique_npcis, rf_param)

        # Compute weights
        W, idx_sort = compute_weights_single(m_rp, idx_rp, m_tp, idx_tp)

        # Estimate location
        location_est, error = wknn_single(test_point.iloc[0], rps, idx_sort, W, k_wknn)
        errors.append(error)
    return errors


def run_experiment(
    df: pd.DataFrame,
    random_seeds: np.ndarray,
    n_runs: int,
    k_wknn: int,
    rf_param: RF_PARAM_5G,
    clustering_rf_param: RF_PARAM_5G,
    cluster_range: range,
    operator_choice: list[int],
    use_beam_matching: bool,
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

    errors_dict = []
    complexity_dict = []
    runtime_dict = []

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
                use_beam_matching,
            )
            for i in range(n_runs)
        ]
        for future in futures:
            errors, complexity, runtime = future.result()
            errors.append(errors)
            complexity.append(complexity)
            runtime.append(runtime)

    errors_df = pd.DataFrame(errors_dict)
    complexity_df = pd.DataFrame(complexity_dict)
    runtime_df = pd.DataFrame(runtime_dict)

    return errors_df, complexity_df, runtime_df


def main():
    # Parameters
    n_runs = 2
    k_wknn = 2
    rf_param = RF_PARAM_5G.RSRQ
    clustering_rf_param = RF_PARAM_5G.RSRQ
    cluster_range = range(0, 1)
    operator_choice = [10]
    selected_campaigns = list(range(1, 11))

    df, random_seeds = load_data(selected_campaigns, rf_param, operator_choice)

    start_time = time.time()

    errors_df, complexity_df, runtime_df = run_experiment(
        df,
        random_seeds,
        n_runs,
        k_wknn,
        rf_param,
        clustering_rf_param,
        cluster_range,
        operator_choice,
        True,
    )

    df, random_seeds = load_data(selected_campaigns, rf_param, operator_choice)

    # run without beam matching as controle
    ctr_errors_df, ctr_complexity_df, ctr_runtime_df = run_experiment(
        df,
        random_seeds,
        n_runs,
        k_wknn,
        rf_param,
        clustering_rf_param,
        cluster_range,
        operator_choice,
        False,
    )

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total runtime was {total_time} seconds")

    print(f"errors", errors_df)

    print(f"control errors", ctr_errors_df)

    config = {
        "wknn_k": k_wknn,
        "rf_param": rf_param.value,
        "cluster_rf_param": clustering_rf_param.value,
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
        "control_errors": ctr_errors_df,
        "control_complexity": ctr_complexity_df,
        "control_runtime": ctr_runtime_df,
    }

    # save_experiment_result("beam_matching_experiment", config, data)


if __name__ == "__main__":
    main()
