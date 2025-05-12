import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from scripts.data_filter import filter_dataframe
from scripts.data_loader import load_dataframe
from scripts.data_writer import save_experiment_result
from scripts.utils import RF_PARAM_5G, NETWORK_TYPE, extract_unique_npcis, get_config
from scripts.weighted_coverage import run_weighted_coverage


def load_data(
    selected_campaigns: list[int],
    rf_param: RF_PARAM_5G,
    operator_choice: list[int],
    freqs: list[int],
) -> pd.DataFrame:
    filename = "5G_data_2023.mat"

    # Series of random seeds for reproducability
    random_seeds = np.loadtxt("config/random_seeds.csv", dtype=int)

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
        freqs=freqs,
    )

    return df, random_seeds


def run_experiment(
    inp_dataframe: pd.DataFrame,
    config: dict,
    random_seeds: np.ndarray,
    threadpool_size: int = 5,
    arfcns: list[int] = None,
):
    # filter data based on n best pcis and beams

    data = []
    means = []

    data_columns = ["error", "complexity", "runtime", "run", "pci_beam_config"]

    operator_configs = [
        [1],  # Single non-beamforming operator
        [10],  # Single (best) beamforming operator
        [1, 10],  # Combination of beamforming and none beamfomring operators
        [10, 50, 88],  # All beamforming operators
        [1, 10, 50, 88],  # All operators
    ]
    for config_id, op_conf in enumerate(operator_configs):
        print("Running operator config", config_id)

        df = filter_dataframe(inp_dataframe.copy(), operators=op_conf)

        unique_pcis = extract_unique_npcis(df["measurements_matrix"])

        with ProcessPoolExecutor(max_workers=threadpool_size) as executor:
            futures = [
                (
                    executor.submit(
                        run_weighted_coverage,
                        df,
                        config["rf_param"],
                        config["rf_param"],
                        config["wknn_k"],
                        unique_pcis,
                        random_seeds[run],
                        config["n_clusters"],
                    ),
                    run,
                )
                for run in range(config["n_runs"])
            ]
            for future, run in futures:
                result, runtime, num_tps = future.result()
                print(f"Run {run} done")
                extra = np.array([runtime, run, config_id])
                extra = np.tile(extra, (result.shape[0], 1))
                res = np.concatenate([result, extra], axis=1)
                # store full results
                data.extend(res)
                # store mean results
                result_mean = result.mean(axis=0)
                means.append([result_mean[0], result_mean[1], runtime, run, config_id])

    results_df = pd.DataFrame(data, columns=data_columns)
    results_df_means = pd.DataFrame(means, columns=data_columns)

    return results_df, results_df_means


def main():

    config = {
        "wknn_k": 2,
        "rf_param": RF_PARAM_5G.RSRQ,
        "operator_choice": [1, 10],
        "n_clusters": 10,
        "n_runs": 20,
        "n_best_pcis": 15,
        "bands": [78],
        "n_best_beams": 4,
        "selected_campaigns": list(range(1, 21)),
        "threadpool_size": 5,
    }

    # find correct nr_arfcn's from band
    band_config = get_config("band_map.json")
    selected_arfcns = [
        int(arfcn)
        for mapping in band_config.values()
        for arfcn, band in mapping.items()
        if band in config["bands"]
    ]
    config["arfcns"] = selected_arfcns

    print(f"ARFCNs: {config['arfcns']}")

    # load data
    df, random_seeds = load_data(
        config["selected_campaigns"],
        config["rf_param"],
        operator_choice=config["operator_choice"],
        freqs=config["arfcns"],
    )

    # time the experiment
    start_time = time.time()

    results_df, results_df_means = run_experiment(
        df,
        config,
        random_seeds,
        threadpool_size=config["threadpool_size"],
    )
    total_time = time.time() - start_time

    print(
        f"Experiment complete in {total_time:.0f} seconds ({total_time/60:.0f} minutes)"
    )

    data = {
        "results": results_df,
        "results_means": results_df_means,
    }

    # fix serializing
    config["rf_param"] = config["rf_param"].value

    save_experiment_result("frequency_operator_experiment", config, data)


if __name__ == "__main__":
    main()
