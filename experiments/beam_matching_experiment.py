import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
from numpy import ndarray, dtype

from scripts.beamforming import (
    get_best_beam,
    get_beam_sidelobe_pcis,
    filter_best_beams,
)
from scripts.data_filter import filter_dataframe
from scripts.data_loader import load_dataframe
from scripts.data_writer import save_experiment_result
from scripts.matrix_operations import create_point_matrix, compute_weights
from scripts.single import wknn_one_tp_row
from scripts.utils import (
    NETWORK_TYPE,
    RF_PARAM_5G,
    extract_unique_npcis,
    dataset_tp_rp_split,
)
from scripts.weighted_coverage import wknn


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


def beam_matching_strategy(
    df: pd.DataFrame,
    rf_param: RF_PARAM_5G,
    random: int,
    run: int,
    use_best_beam: bool,
    use_sidelobes: bool,
    n_best_pcis: int,
) -> ndarray[Any, dtype[Any]]:
    # get the best beam for each point
    df["best_beam"] = df["measurements_matrix"].apply(
        lambda x: get_best_beam(x, rf_param)
    )
    print(
        f"RUNNING STRATEGY -> SIDELOBES: {use_sidelobes} -> BEST BEAM {use_best_beam}"
    )
    unique_npcis = extract_unique_npcis(df["measurements_matrix"])

    # Pre-compute the control matrix (all RPs) once
    df_tp, df_rp = dataset_tp_rp_split(df, 0.3, random)

    m_rp_control, idx_rp_control = create_point_matrix(df_rp, unique_npcis, rf_param)
    m_tp_control, idx_tp_control = create_point_matrix(df_tp, unique_npcis, rf_param)
    W_control, idx_sort_control = compute_weights(
        m_rp_control, idx_rp_control, m_tp_control, idx_tp_control
    )
    _, errors_control = wknn(df_tp, df_rp, idx_sort_control, W_control, k=2)
    complexity_control = m_rp_control.shape[0] * m_rp_control.shape[1]

    # Pre-compute reference point matrices by beam
    rp_matrices_by_beam = {}
    unique_beams = df_rp["best_beam"].unique()

    # Pre-compute matrices for each beam group
    for beam in unique_beams:
        beam_rps = df_rp[df_rp["best_beam"] == beam]

        pcis = unique_npcis
        if use_best_beam:
            if use_sidelobes:
                pcis = get_beam_sidelobe_pcis(beam)
            else:
                pcis = [beam]

        if use_best_beam:
            beam_rps_filtered = beam_rps.copy()

            beam_rps_filtered["measurements_matrix"] = beam_rps[
                "measurements_matrix"
            ].apply(
                lambda x: (
                    filter_best_beams(
                        x,
                        rf_param,
                        n_best_pcis=n_best_pcis,
                        use_sidelobes=use_sidelobes,
                    )
                )
            )
            if n_best_pcis > 1:
                pcis = extract_unique_npcis(beam_rps_filtered["measurements_matrix"])

            m_rp, idx_rp = create_point_matrix(beam_rps_filtered, pcis, rf_param)
        else:
            m_rp, idx_rp = create_point_matrix(beam_rps, pcis, rf_param)

        rp_matrices_by_beam[beam] = (m_rp, idx_rp, beam_rps, pcis)

    data = []

    for i, (_, tp_row) in enumerate(df_tp.iterrows(), 1):
        tp = pd.DataFrame([tp_row])
        best_beam = tp_row["best_beam"]

        # Get the pre-computed matrices for this beam
        if best_beam in rp_matrices_by_beam:
            m_rp, idx_rp, rps, pcis_tp = rp_matrices_by_beam[best_beam]

        else:
            continue

        # Create the point matrix for the test point
        m_tp, idx_tp = create_point_matrix(tp, pcis_tp, rf_param)

        # Compute weights only if we have matching RPs

        W, idx_sort = compute_weights(m_rp, idx_rp, m_tp, idx_tp)
        _, err = wknn_one_tp_row(tp, rps, idx_sort, W, 2)

        complexity = m_rp.shape[0] * m_rp.shape[1]

        data.append(
            [
                err,
                complexity,
                str(best_beam),
                tp_row["lat"],
                tp_row["lng"],
            ]
        )

    print(f"RUN {run} completed \t PID: {os.getpid()}")

    data = np.array(data)

    print(data.shape)
    print(data)

    return data


def run_experiment(
    df: pd.DataFrame,
    random_seeds: np.ndarray,
    n_runs: int,
    rf_param: RF_PARAM_5G,
    use_best_beam: bool,
    use_sidelobes: bool,
    n_best_pcis: int,
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
            executor.submit(
                beam_matching_strategy,
                df,
                rf_param,
                random_seeds[i],
                i,
                use_best_beam,
                use_sidelobes,
                n_best_pcis,
            )
            for i in range(n_runs)
        ]
        for future in futures:
            res = future.result()
            data.extend(res)

    data_df = pd.DataFrame(
        data,
        columns=[
            "errors",
            "complexity",
            "best_beam",
            "lat",
            "lng",
        ],
    )

    return data_df


def main():
    # Parameters
    n_runs = 1
    k_wknn = 2
    rf_param = RF_PARAM_5G.RSRQ
    operator_choice = [10]
    selected_campaigns = list(range(1, 11))

    # load the data

    df, random_seeds = load_data(selected_campaigns, rf_param, operator_choice)

    start_time = time.time()

    config = [
        {
            "label": "baseline",
            "use_best_beam": False,
            "use_sidelobes": False,
            "n_best_pcis": 1,
        },
        # {
        #     "label": "best_beam",
        #     "use_best_beam": True,
        #     "use_sidelobes": False,
        #     "n_best_pcis": 1,
        # },
        # {
        #     "label": "best_beam_with_sidelobes",
        #     "use_best_beam": True,
        #     "use_sidelobes": True,
        #     "n_best_pcis": 1,
        # },
    ]

    results = {}

    results_df = pd.DataFrame()

    for conf in config:
        print(f"RUN {conf['label']}")

        # baseline measurement
        data_df = run_experiment(
            df.copy(deep=True),
            random_seeds,
            n_runs,
            rf_param,
            use_best_beam=conf["use_best_beam"],
            use_sidelobes=conf["use_sidelobes"],
            n_best_pcis=conf["n_best_pcis"],
        )

        data_df["use_best_beam"] = conf["use_best_beam"]
        data_df["use_sidelobes"] = conf["use_sidelobes"]
        data_df["n_best_pcis"] = conf["n_best_pcis"]

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

    print(results_df.head(10))
    data = {"data": results_df}

    save_experiment_result("beam_matching_experiment", config, data)


if __name__ == "__main__":
    main()
