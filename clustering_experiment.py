import time

import numpy as np
import pandas as pd
from tabulate import tabulate

from scripts.data_filter import filter_dataframe
from scripts.data_loader import load_dataframe
from scripts.data_writer import save_experiment_result
from scripts.utils import NETWORK_TYPE
from scripts.utils import extract_unique_npcis, RF_PARAM_5G
from scripts.weighted_coverage import run_weighted_coverage


def clustering_experiment():
    """
    Parameters
    """
    n_runs = 5
    k_wknn = 2
    rf_params = [RF_PARAM_5G.RSRQ]
    clustering_params = [RF_PARAM_5G.RSRQ]
    operator_choice = [10]
    cluster_range = range(2, 4)
    campaigns = [1, 2, 3, 4, 5, 6, 8, 9, 10]

    # Loading the data

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

    print(f"Loaded dataframe with {len(df)} rows")

    df = filter_dataframe(
        df=df,
        operators=[10],
        include_columns=["pci", "beam_index", "nr_arfcn", "operator_id", "rsrq"],
        campaigns=[1, 2, 3, 4, 5, 6],
    )

    print(f"Filtered dataframe. now  {len(df)} rows")

    print(
        f"""
    CLUSTERING EXPERIMENT
    
    🧪 Experiment setup 🧪
    🔢 k-value for wKNN = {k_wknn}
    👨‍👩‍👦‍👦 cluster range = {cluster_range[0]} - {cluster_range[-1]}
    🛜 RF PARAM {str(rf_params)}
    📶 Operator choice {operator_choice}
    🔁 Number of runs {n_runs}
    🤵‍♂️ Campaigns {campaigns}
    _________________________________
    """
    )

    unique_npcis = extract_unique_npcis(df["measurements_matrix"])
    errors_dict = {c: [] for c in cluster_range}  # Store the errors
    complexity_dict = {c: [] for c in cluster_range}
    runtime_dict = {c: [] for c in cluster_range}

    start_time = time.time()

    for c in cluster_range:
        for i in range(n_runs):
            print(f"\r🔄 Running for {c} clusters ({i + 1}/{n_runs} runs)", end="")
            random = random_seeds[i]

            _, errors, complexity, runtime = run_weighted_coverage(
                df=df,
                rf_params=rf_params,
                cluster_rf_params=clustering_params,
                k_max=k_wknn,
                unique_npcis=unique_npcis,
                random_seed=random,
                n_clusters=c,
            )
            errors_dict[c].append(errors.mean())
            complexity_dict[c].append(complexity)
            runtime_dict[c].append(runtime)

        print(f"\r✅ {c} completed                                           ")

    errors_df = pd.DataFrame(errors_dict)
    complexity_df = pd.DataFrame(complexity_dict)
    runtime_df = pd.DataFrame(runtime_dict)

    end_time = time.time()

    print(f"Completed in {end_time - start_time} seconds")

    config = {
        "wknn_k": k_wknn,
        "rf_param": rf_params[0].value,
        "cluster_rf_param": clustering_params[0].value,
        "operator_choice": operator_choice,
        "cluster_range": list(cluster_range),
        "campaigns": campaigns,
        "n_runs": n_runs,
    }

    data = {
        "errors": errors_df,
        "complexity": complexity_df,
        "runtime": runtime_df,
    }

    # Assuming errors_df is your DataFrame
    table_data = [[col, errors_df[col].mean()] for col in errors_df.columns]

    # Print the table
    print(tabulate(table_data, headers=["K", "Error"], tablefmt="grid"))

    save_experiment_result("clustering-experiment", config, data)


if __name__ == "__main__":
    clustering_experiment()
