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
    get_config,
    replace_nr_arfcns,
    extract_unique_npcis,
)
from scripts.weighted_coverage import run_weighted_coverage


def load_data(
    selected_campaigns: list[int],
    rf_param: RF_PARAM_5G,
    operator_choice: list[int],
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
        operators=operator_choice,
        campaigns=selected_campaigns,
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
    nr,
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
    print(f"🔄 Running for nr_arfcn {nr} ({i + 1}/{n_runs} runs) on PID: {os.getpid()}")
    result, runtime, num_tps = run_weighted_coverage(
        df=filtered_df,
        rf_param=rf_param,
        cluster_rf_param=rf_param,
        k_max=k_wknn,
        unique_npcis=unique_npcis,
        random_seed=random_seed,
        n_clusters=n_clusters,
        use_pca=False,
    )
    return result, runtime, num_tps


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

    config = get_config("frequency_map.json", str(operator_choice[0]))
    nr_arfcn_frequecny_map = {int(k): int(v) for k, v in config.items()}

    print(nr_arfcn_frequecny_map)

    frequency_choice = list(set(nr_arfcn_frequecny_map.values())) + [0]
    print("frequency_choice", frequency_choice)
    replace_nr_arfcns(df, nr_arfcn_frequecny_map)

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

    results = []

    for op in operator_choice:
        tmp = filter_dataframe(df=df.copy(), operators=[op])

        config = get_config("frequency_map.json", str(op))
        nr_arfcn_frequecny_map = {int(k): int(v) for k, v in config.items()}
        frequency_choice = list(set(nr_arfcn_frequecny_map.values())) + [0]
        replace_nr_arfcns(tmp, nr_arfcn_frequecny_map)
        for nr in frequency_choice:
            if nr != 0:
                filtered_df = filter_dataframe(df=tmp.copy(), freqs=[nr])
                print(f"num items after filter {len(filtered_df)}")
            else:
                filtered_df = tmp.copy()

            unique_npcis = extract_unique_npcis(filtered_df["measurements_matrix"])

            # Use ProcessPoolExecutor to parallelize the runs
            with ProcessPoolExecutor(max_workers=1) as executor:
                futures = [
                    executor.submit(
                        single_run,
                        nr,
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
                    data, runtime, num_tps = future.result()
                    # extra = np.array([op, nr, runtime])
                    # extra = np.tile(extra, (data.shape[0], 1))
                    # res = np.concatenate([extra, data], axis=1)
                    # results.extend(res)

                    # dont store all TPs, only mean over runs
                    means = data.mean(axis=0)
                    results.append([op, nr, runtime, num_tps, means[0], means[1]])
            print(f"\r✅ {nr} completed                                           ")

    results_df = pd.DataFrame(
        results,
        columns=["operator", "frequency", "runtime", "num_tps", "error", "complexity"],
    )

    return results_df


def main():
    # Parameters
    n_runs = 20
    k_wknn = 2
    rf_param = RF_PARAM_5G.RSRQ
    clustering_rf_param = RF_PARAM_5G.RSRQ
    n_clusters = 10
    operator_choice = [1, 10, 50, 88]
    selected_campaigns = list(range(0, 21))
    use_best_beams = False

    start_time = time.time()

    # vodafone
    df, random_seeds = load_data(selected_campaigns, rf_param, operator_choice)

    results_df = run_experiment(
        df,
        random_seeds,
        n_runs,
        k_wknn,
        rf_param,
        clustering_rf_param,
        n_clusters,
        operator_choice,
    )

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

    data = {"data": results_df}

    save_experiment_result("frequency_experiment", config, data)


if __name__ == "__main__":
    main()
