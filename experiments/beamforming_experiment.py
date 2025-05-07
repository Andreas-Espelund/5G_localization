import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from scripts.beamforming import matrix_filter
from scripts.data_filter import filter_dataframe
from scripts.data_loader import load_dataframe
from scripts.data_writer import save_experiment_result
from scripts.utils import (
    NETWORK_TYPE,
    RF_PARAM_5G,
    extract_unique_npcis,
    get_config,
)
from scripts.weighted_coverage import run_weighted_coverage


def load_data(
    selected_campaigns: list[int],
    rf_param: RF_PARAM_5G,
    operator_choice: list[int],
    bands: list[int],
) -> pd.DataFrame:
    filename = "5G_data_2023.mat"

    # Series of random seeds for reproducability
    random_seeds = np.loadtxt("../data/random_seeds.csv", dtype=int)

    # load the dataframe from saved file or 'raw' matlab file
    df = load_dataframe(filename, NETWORK_TYPE._5G)

    # # Drop unused columns to save space
    matrix_cols_to_drop = ["toa_pps", "toa_cir", "toa_cov", "campaign_id"]
    df["measurements_matrix"] = df["measurements_matrix"].apply(
        lambda x: x.drop(columns=matrix_cols_to_drop)
    )

    band_config = get_config("band_map.json")

    selected_arfcns = [
        int(arfcn)
        for mapping in band_config.values()
        for arfcn, band in mapping.items()
        if band in bands
    ]

    # Data filtering
    df = filter_dataframe(
        df=df,
        operators=operator_choice,
        campaigns=selected_campaigns,
        freqs=selected_arfcns,
        include_columns=[
            "pci",
            "beam_index",
            "nr_arfcn",
            "operator_id",
            rf_param.value,
        ],
    )

    return df, random_seeds


def single_run(
    i, filtered_df, rf_param, unique_npcis, random_seed, n_clusters, k_wknn, n_runs
):
    print(
        f"🔄 Running for cluster {n_clusters} ({i + 1}/{n_runs} runs) on PID: {os.getpid()}"
    )
    result, runtime, n_tps = run_weighted_coverage(
        df=filtered_df,
        rf_param=rf_param,
        cluster_rf_param=rf_param,
        k_max=k_wknn,
        unique_npcis=unique_npcis,
        random_seed=random_seed,
        n_clusters=n_clusters,
    )

    return result, runtime, n_tps, i


def run_experiment_operators(
    df_orig: pd.DataFrame,
    random_seeds: np.ndarray,
    n_runs: int,
    rf_param: RF_PARAM_5G,
    n_best_pcis: int,
    n_best_beams: int,
    wknn_k: int,
    n_clusters: int,
):
    if n_best_pcis:
        n_best_pcis = int(n_best_pcis)
    if n_best_beams:
        n_best_beams = int(n_best_beams)

    data = []

    for op in [1, 10, 50, 88]:
        df = filter_dataframe(df_orig.copy(deep=True), operators=[op])

        df.loc[:, "measurements_matrix"] = df.loc[:, "measurements_matrix"].apply(
            lambda x: matrix_filter(
                x,
                rf_param,
                include_n_best_pcis=n_best_pcis,
                include_n_best_beams=n_best_beams,
            )
        )

        unique_npcis = extract_unique_npcis(df["measurements_matrix"])

        # Use ProcessPoolExecutor to parallelize the runs
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [
                executor.submit(
                    single_run,
                    i,
                    df,
                    rf_param,
                    unique_npcis,
                    random_seeds[i],
                    n_clusters,
                    wknn_k,
                    n_runs,
                )
                for i in range(n_runs)
            ]
            for future in futures:
                result, runtime, n_tps, run = future.result()

                extra = np.array([runtime, n_tps, op, run])
                extra = np.tile(extra, (result.shape[0], 1))
                res = np.concatenate([result, extra], axis=1)

                data.extend(res)

    data_df = pd.DataFrame(
        data,
        columns=["errors", "complexity", "runtime", "n_tps", "mnc", "run"],
    )

    return data_df


def run_experiment(
    df: pd.DataFrame,
    random_seeds: np.ndarray,
    n_runs: int,
    rf_param: RF_PARAM_5G,
    n_best_pcis: int,
    n_best_beams: int,
    wknn_k: int,
    n_clusters: int,
):
    if n_best_pcis:
        n_best_pcis = int(n_best_pcis)
    if n_best_beams:
        n_best_beams = int(n_best_beams)

    data = []

    df.loc[:, "measurements_matrix"] = df.loc[:, "measurements_matrix"].apply(
        lambda x: matrix_filter(
            x,
            rf_param,
            include_n_best_pcis=n_best_pcis,
            include_n_best_beams=n_best_beams,
        )
    )

    unique_npcis = extract_unique_npcis(df["measurements_matrix"])

    # Use ProcessPoolExecutor to parallelize the runs
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(
                single_run,
                i,
                df,
                rf_param,
                unique_npcis,
                random_seeds[i],
                n_clusters,
                wknn_k,
                n_runs,
            )
            for i in range(n_runs)
        ]
        for future in futures:
            result, runtime, n_tps, run = future.result()

            extra = np.array([runtime, n_tps, 0, run])
            extra = np.tile(extra, (result.shape[0], 1))
            res = np.concatenate([result, extra], axis=1)

            data.extend(res)

    data_df = pd.DataFrame(
        data,
        columns=["errors", "complexity", "runtime", "n_tps", "mnc", "run"],
    )

    return data_df


def main():
    # Parameters
    n_runs = 20
    k_wknn = 2
    n_clusters = 10
    rf_param = RF_PARAM_5G.RSRQ
    operator_choice = [1, 10, 50, 88]
    selected_campaigns = list(range(1, 21))
    bands = [78]

    # load the data

    df, random_seeds = load_data(selected_campaigns, rf_param, operator_choice, bands)

    start_time = time.time()

    # Basic configuration
    config_params = []

    pci_configs = range(1, 21)
    beam_configs = range(1, 9)

    # pci_configs = [1, None]
    # beam_configs = [1, None]

    for n_pcis in pci_configs:
        for n_beams in beam_configs:
            config_params.append([n_pcis, n_beams])

    # for n_beams in beam_configs:
    #     config_params.append([None, n_beams])

    results_df = pd.DataFrame()

    total_configs = len(config_params)

    for index, conf in enumerate(config_params):
        n_best_pcis = conf[0]
        n_best_beams = conf[1]
        # baseline measurement

        run_start = time.time()
        print(f"🔄 Running confing {index + 1} / {total_configs}")

        tmp = df.copy(deep=True)

        data_df = run_experiment(
            tmp,
            random_seeds,
            n_runs,
            rf_param,
            n_best_pcis=n_best_pcis,
            n_best_beams=n_best_beams,
            n_clusters=n_clusters,
            wknn_k=k_wknn,
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

    print(f"Total runtime was {total_time} seconds ({total_time / 60} minutes)")

    config = {
        "wknn_k": k_wknn,
        "rf_param": rf_param.value,
        "operator_choice": operator_choice,
        "n_runs": n_runs,
        "campaigns": selected_campaigns,
        "n_clusters": n_clusters,
        "runtime": total_time,
    }

    data = {"data": results_df}

    save_experiment_result("new_beam_matching_experiment", config, data)


if __name__ == "__main__":
    main()
