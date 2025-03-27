import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from scripts.beamforming import get_best_beam, matrix_filter
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
from scripts.weighted_coverage import wknn_one_tp_row, wknn_one


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


def beam_matching_strategy_2(
    df: pd.DataFrame,
    rf_param: RF_PARAM_5G,
    random: int,
    run: int,
    n_best_pcis: int,
    n_best_beams: int,
) -> np.ndarray:

    df = df.sample(frac=1, random_state=random).reset_index(drop=True)

    df.loc[:, "measurements_matrix"] = df.loc[:, "measurements_matrix"].apply(
        lambda x: matrix_filter(
            x,
            rf_param,
            include_n_best_pcis=n_best_pcis,
            include_n_best_beams=n_best_beams,
        )
    )

    pcis = extract_unique_npcis(df["measurements_matrix"])

    df_tp, df_rp = dataset_tp_rp_split(df, 0.3, random)

    m_rp, idx_rp = create_point_matrix(df_rp, pcis, rf_param)

    m_tp, idx_tp = create_point_matrix(df_tp, pcis, rf_param)

    W, idx_sort = compute_weights(m_rp, idx_rp, m_tp, idx_tp)

    _, errors = wknn_one(df_tp, df_rp, idx_sort, W, 2)

    complexity = m_rp.shape[0] * m_rp.shape[1]

    data = np.array(
        [
            errors,
            np.repeat(complexity, errors.shape[0]),
            np.repeat(run, errors.shape[0]),
        ]
    )
    print(f"Run {run} complete")
    return data.T


def beam_matching_strategy(
    df: pd.DataFrame,
    rf_param: RF_PARAM_5G,
    random: int,
    run: int,
    n_best_pcis: int,
    n_best_beams: int,
) -> np.ndarray:
    # get the best beam for each point
    df["best_beam"] = df["measurements_matrix"].apply(
        lambda x: get_best_beam(x, rf_param)
    )

    unique_npcis = extract_unique_npcis(df["measurements_matrix"])

    df_tp, df_rp = dataset_tp_rp_split(df, 0.3, random)

    df_rp.loc[:, "measurements_matrix"] = df_rp.loc[:, "measurements_matrix"].apply(
        lambda x: matrix_filter(
            x,
            rf_param,
            include_n_best_pcis=n_best_pcis,
            include_n_best_beams=n_best_beams,
        )
    )

    m_rp, idx_rp = create_point_matrix(df_rp, unique_npcis, rf_param)

    data = []
    total = len(df_tp)
    for i, (_, tp_row) in enumerate(df_tp.iterrows(), 1):
        tp = pd.DataFrame([tp_row])

        tp.loc[:, "measurements_matrix"] = tp.loc[:, "measurements_matrix"].apply(
            lambda x: matrix_filter(
                x,
                rf_param,
                include_n_best_pcis=n_best_pcis,
                include_n_best_beams=n_best_beams,
            )
        )

        # Create the point matrix for the test point
        m_tp, idx_tp = create_point_matrix(tp, unique_npcis, rf_param)

        # Compute weights only if we have matching RPs
        W, idx_sort = compute_weights(m_rp, idx_rp, m_tp, idx_tp)
        _, errors = wknn_one_tp_row(tp, df_rp, idx_sort, W, 2)

        complexity = m_rp.shape[0] * m_rp.shape[1] if len(m_rp.shape) == 2 else None

        data.append([errors, complexity, run])

    data = np.array(data)
    print(f"\tRun {run} complete")
    return data


def run_experiment(
    df: pd.DataFrame,
    random_seeds: np.ndarray,
    n_runs: int,
    rf_param: RF_PARAM_5G,
    n_best_pcis: int,
    n_best_beams: int,
):
    if n_best_pcis:
        n_best_pcis = int(n_best_pcis)
    if n_best_beams:
        n_best_beams = int(n_best_beams)

    data = []

    num_processors = os.cpu_count()
    # Use ProcessPoolExecutor to parallelize the runs
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                beam_matching_strategy_2,
                df,
                rf_param,
                random_seeds[i],
                i,
                n_best_pcis,
                n_best_beams,
            )
            for i in range(n_runs)
        ]
        for future in futures:
            res = future.result()
            data.extend(res)

    data_df = pd.DataFrame(
        data,
        columns=["errors", "complexity", "run"],
    )

    return data_df


def main():
    # Parameters
    n_runs = 10
    k_wknn = 2
    rf_param = RF_PARAM_5G.RSRQ
    operator_choice = [10, 50, 88]
    selected_campaigns = list(range(1, 21))

    # load the data

    df, random_seeds = load_data(selected_campaigns, rf_param, operator_choice)

    start_time = time.time()

    # Basic configuration
    config_params = [[None, None]]

    pci_config = list(range(1, 30))
    # pci_config = [None]
    # beam_config = [1, 2, 3, 4, 5, 6, 7, 8]
    beam_config = [None]

    # pci_config = [20]

    for n_pcis in pci_config:
        for n_beams in beam_config:
            config_params.append([n_pcis, n_beams])

    results_df = pd.DataFrame()

    total_configs = len(config_params)

    for index, conf in enumerate(config_params):
        n_best_pcis = conf[0]
        n_best_beams = conf[1]
        # baseline measurement

        run_start = time.time()
        print(f"🔄 Running confing {index + 1} / {total_configs}")
        data_df = run_experiment(
            df.copy(deep=True),
            random_seeds,
            n_runs,
            rf_param,
            n_best_pcis=n_best_pcis,
            n_best_beams=n_best_beams,
        )
        run_end = time.time()
        print(
            f"✅ Config {index + 1} / {total_configs} done in {run_end - run_start} seconds"
        )

        data_df["n_best_pcis"] = n_best_pcis
        data_df["n_best_beams"] = n_best_beams

        results_df = pd.concat([results_df, data_df], ignore_index=True, axis=0)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total runtime was {total_time} seconds")

    config = {
        "wknn_k": k_wknn,
        "rf_param": rf_param.value,
        "operator_choice": operator_choice,
        "n_runs": n_runs,
        "campaigns": selected_campaigns,
        "runtime": total_time,
    }

    print("\n================ RESULTS ================\n")
    print(results_df["errors"].to_numpy())

    data = {"data": results_df}

    save_experiment_result("new_beam_matching_experiment", config, data)


if __name__ == "__main__":
    main()
