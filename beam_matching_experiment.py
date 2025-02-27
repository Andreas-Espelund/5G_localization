import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

import numpy as np
import pandas as pd

from scripts.beamforming import get_best_beam
from scripts.data_filter import filter_dataframe
from scripts.data_loader import load_dataframe
from scripts.data_writer import save_experiment_result
from scripts.matrix_operations import create_point_matrix, compute_weights
from scripts.utils import (
    NETWORK_TYPE,
    RF_PARAM_5G,
    extract_unique_npcis,
    dataset_tp_rp_split,
)
from scripts.weighted_coverage import wknn_one_tp_row


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


def beam_matching_strategy(
    df: pd.DataFrame, rf_param: RF_PARAM_5G, random: int, run: int
) -> Tuple[float, float, Tuple[int, int], Tuple[int, int]]:
    # get the best beam for each point
    df["best_beam"] = df["measurements_matrix"].apply(
        lambda x: get_best_beam(x, rf_param)
    )

    unique_npcis = extract_unique_npcis(df["measurements_matrix"])

    df_tp, df_rp = dataset_tp_rp_split(df, 0.3, random)

    # Pre-compute reference point matrices by beam
    rp_matrices_by_beam = {}
    unique_beams = df_rp["best_beam"].unique()

    # Pre-compute matrices for each beam group
    for beam in unique_beams:
        beam_rps = df_rp[df_rp["best_beam"] == beam]
        m_rp, idx_rp = create_point_matrix(beam_rps, unique_npcis, rf_param)
        rp_matrices_by_beam[beam] = (m_rp, idx_rp, beam_rps)

    # Pre-compute the control matrix (all RPs) once
    m_rp_control, idx_rp_control = create_point_matrix(df_rp, unique_npcis, rf_param)

    data = []

    for i, (_, tp_row) in enumerate(df_tp.iterrows(), 1):
        tp = pd.DataFrame([tp_row])
        best_beam = tp_row["best_beam"]

        # Get the pre-computed matrices for this beam
        if best_beam in rp_matrices_by_beam:
            m_rp, idx_rp, rps = rp_matrices_by_beam[best_beam]
        else:
            # Handle the case where the beam isn't in reference points
            rps = pd.DataFrame()  # Empty DataFrame
            m_rp, idx_rp = np.array([]), np.array([])

        # Create the point matrix for the test point
        m_tp, idx_tp = create_point_matrix(tp, unique_npcis, rf_param)

        # Compute weights only if we have matching RPs
        if len(rps) > 0:
            W, idx_sort = compute_weights(m_rp, idx_rp, m_tp, idx_tp)
            _, errors = wknn_one_tp_row(tp, rps, idx_sort, W, 2)
        else:
            continue

        # Compute control weights and errors
        W_control, idx_sort_control = compute_weights(
            m_rp_control, idx_rp_control, m_tp, idx_tp
        )
        _, errors_control = wknn_one_tp_row(tp, df_rp, idx_sort_control, W_control, 2)

        complexity = m_rp.shape[0] * m_rp.shape[1] if len(m_rp.shape) == 2 else None
        complexity_control = (
            m_rp_control.shape[0] * m_rp_control.shape[1]
            if len(m_rp_control.shape) == 2
            else None
        )
        data.append([errors, errors_control, complexity, complexity_control])

    print(f"RUN {run} completed\r PID: {os.getpid()}")
    data = np.array(data)
    return data.mean(axis=0)


def run_experiment(
    df: pd.DataFrame,
    random_seeds: np.ndarray,
    n_runs: int,
    rf_param: RF_PARAM_5G,
):

    data = []

    # find the best beams for each tp for later matching between TP and RP
    df["best_beam"] = df["measurements_matrix"].apply(
        lambda x: get_best_beam(x, rf_param)
    )

    num_processors = os.cpu_count()
    # Use ProcessPoolExecutor to parallelize the runs
    with ProcessPoolExecutor(max_workers=num_processors) as executor:
        futures = [
            executor.submit(beam_matching_strategy, df, rf_param, random_seeds[i], i)
            for i in range(n_runs)
        ]
        for future in futures:
            res = future.result()
            data.append(res)

    data_df = pd.DataFrame(
        data, columns=["errors", "errors_control", "complexity", "complexity_control"]
    )

    return data_df


def main():
    # Parameters
    n_runs = 20
    k_wknn = 2
    rf_param = RF_PARAM_5G.RSRQ
    operator_choice = [10]
    selected_campaigns = list(range(1, 41))

    # load the data

    print("LOADING DATA")
    df, random_seeds = load_data(selected_campaigns, rf_param, operator_choice)

    start_time = time.time()

    print(f"RUNNING EXPERIMENT")
    data_df = run_experiment(
        df,
        random_seeds,
        n_runs,
        rf_param,
    )

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total runtime was {total_time} seconds")
    print(f"errors", data_df)

    config = {
        "wknn_k": k_wknn,
        "rf_param": rf_param.value,
        "operator_choice": operator_choice,
        "n_runs": n_runs,
        "campaigns": selected_campaigns,
        "runtime": total_time,
    }

    data = {
        "data": data_df,
    }

    save_experiment_result("beam_matching_experiment", config, data)


if __name__ == "__main__":
    main()
