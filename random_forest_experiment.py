import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

from scripts.clustering import train_random_forest
from scripts.data_filter import filter_dataframe
from scripts.data_loader import load_dataframe
from scripts.data_writer import save_experiment_result
from scripts.matrix_operations import create_point_matrix
from scripts.utils import NETWORK_TYPE, extract_unique_npcis, RF_PARAM_5G
from scripts.utils import dataset_tp_rp_split


def evaluate_model(rf_model, X_test, y_test):
    # Predict the test set
    y_pred = rf_model.predict(X_test)

    # Calculate overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    cv_scores = cross_val_score(rf_model, X_test, y_test, cv=5)
    mean_cv_score = cv_scores.mean()

    # Create a dictionary of metrics
    metrics = {
        "accuracy": accuracy,
        "recall": recall,
        "f1_score": f1,
        "mean_cv_score": mean_cv_score,
    }

    return metrics


def run_test(df: pd.DataFrame, n_clusters: int, rf_param: RF_PARAM_5G, random: int):

    kmeans = KMeans(n_clusters=n_clusters, random_state=random)
    df_features = df[["lat", "lng"]].values
    cluster_labels = kmeans.fit_predict(df_features)

    df["cluster"] = cluster_labels
    pcis = extract_unique_npcis(df["measurements_matrix"])

    df_tp, df_rp = dataset_tp_rp_split(df, 0.3, random)

    rf_model = train_random_forest(
        df=df_rp,
        unique_npcis=pcis,
        rf_param=rf_param,
        n_estimators=100,
        random_state=random,
    )

    tp_features, _ = create_point_matrix(df_tp, pcis, rf_param)
    df_tp["predicted_cluster"] = rf_model.predict(tp_features)

    # Prepare test data
    y_test = df_tp["cluster"]
    X_test = tp_features

    # Evaluate model and return metrics
    metrics = evaluate_model(rf_model, X_test, y_test)
    return metrics


def main():

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

    selected_campaigns = list(range(1, 20))
    # Data filtering
    df = filter_dataframe(
        df=df,
        include_columns=[
            "pci",
            "beam_index",
            "nr_arfcn",
            "operator_id",
            "rsrq",
            "sinr",
            "rssi",
            "rsrp",
        ],
        campaigns=selected_campaigns,
    )

    # params = [RF_PARAM_5G.RSRQ, RF_PARAM_5G.RSSI, RF_PARAM_5G.RSRP, RF_PARAM_5G.SINR]
    params = [RF_PARAM_5G.RSRQ]
    cluster_range = range(2, 21)
    n_runs = 1
    data = []

    with ThreadPoolExecutor(max_workers=n_runs) as executor:
        futures = []
        for param in params:
            for n_clusters in cluster_range:
                for run in range(n_runs):
                    start = time.time()

                    future = executor.submit(
                        run_test,
                        df=df,
                        n_clusters=n_clusters,
                        rf_param=param.RSRQ,
                        random=random_seeds[n_clusters + run],
                    )
                    futures.append((future, param, n_clusters, run, start))

        for future, param, n_clusters, run, start in futures:
            res = future.result()
            res["param"] = param.value
            res["n_clusters"] = n_clusters
            res["run"] = run
            data.append(res)
            end = time.time()
            print(
                f"Parameter: {param.value}, n_clusters: {n_clusters} run: {run+1}/ {n_runs}  [{end-start:.0f} s]"
            )

    # Create a DataFrame from the collected data
    res_df = pd.DataFrame(data)

    config = {
        "params": [p.value for p in params],
        "cluster_range": list(cluster_range),
        "n_runs": n_runs,
        "selected_campaigns": selected_campaigns,
    }
    data = {"data": res_df}

    print(res_df)

    save_experiment_result("random_forest_experiment", config, data)


if __name__ == "__main__":
    main()
