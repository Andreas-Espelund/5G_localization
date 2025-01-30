import time

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


def load_data(selected_campaigns: list[int]):
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
        include_columns=["pci", "beam_index", "nr_arfcn", "operator_id", "rsrq"],
        campaigns=selected_campaigns,
    )

    return df, random_seeds


def run_experiment(
    df: pd.DataFrame,
    random_seeds: np.ndarray,
    n_runs: int,
    k_wknn: int,
    rf_params: list[RF_PARAM_5G],
    clustering_rf_params: list[RF_PARAM_5G],
    n_clusters: int,
    operator_choice: list[int],
):

    config = get_config("frequency_map.json", "10")
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
    🛜 RF PARAM {str(rf_params)}
    📡 Cluster RF PARAM {str(clustering_rf_params)}
    📶 Operator choice {operator_choice}
    🔁 Number of runs {n_runs}|
    _________________________________
    """
    )

    errors_dict = {nr: [] for nr in frequency_choice}  # Store the errors
    num_entries_dict = {}

    for nr in frequency_choice:
        if nr != 0:
            filtered_df = filter_dataframe(df=df.copy(), freqs=[nr])
        else:
            filtered_df = df.copy()

        unique_npcis = extract_unique_npcis(filtered_df["measurements_matrix"])

        num_entries_dict[nr] = [len(unique_npcis)]

        for i in range(n_runs):
            print(f"\r🔄 Running for nr_arfcn {nr} ({i + 1}/{n_runs} runs)", end="")
            _, errors, _, _ = run_weighted_coverage(
                df=filtered_df,
                rf_params=rf_params,
                cluster_rf_params=rf_params,
                k_max=k_wknn,
                unique_npcis=unique_npcis,
                random_seed=random_seeds[i],
                n_clusters=n_clusters,
            )
            errors_dict[nr].append(errors.mean())

        print(f"\r✅ {nr} completed                                           ")

    errors_df = pd.DataFrame(errors_dict)
    entries_df = pd.DataFrame(num_entries_dict)

    return errors_df, entries_df, frequency_choice


def main():
    # Parameters
    n_runs = 50
    k_wknn = 2
    rf_params = [RF_PARAM_5G.RSRQ]
    clustering_rf_params = [RF_PARAM_5G.RSRQ]
    n_clusters = 5
    operator_choice = [10]
    selected_campaigns = list(range(1, 20))

    df, random_seeds = load_data(selected_campaigns)

    start_time = time.time()

    errors_df, entries_df, frequency_choice = run_experiment(
        df,
        random_seeds,
        n_runs,
        k_wknn,
        rf_params,
        clustering_rf_params,
        n_clusters,
        operator_choice,
    )

    end_time = time.time()

    print(f"Total runtime was {end_time - start_time} seconds")

    config = {
        "wknn_k": k_wknn,
        "rf_param": rf_params[0].value,
        "cluster_rf_param": clustering_rf_params[0].value,
        "operator_choice": operator_choice,
        "nr_arfcn_choice": frequency_choice,
        "n_clusters": n_clusters,
        "n_runs": n_runs,
        "campaigns": selected_campaigns,
    }

    data = {
        "errors": errors_df,
        "entries": entries_df,
    }

    save_experiment_result("frequency_experiment", config, data)


if __name__ == "__main__":
    main()
