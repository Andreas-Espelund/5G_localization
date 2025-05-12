from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from tabulate import tabulate

from scripts.matrix_operations import create_point_matrix, compute_weights
from scripts.utils import RF_PARAM_5G, extract_unique_npcis, haversine_distance
from scripts.weighted_coverage import wknn


class LocalizationModel:
    """
    Object wrapper for using the localization model
    """

    def __init__(
        self,
        n_clusters: int = 10,
        rf_param: RF_PARAM_5G = RF_PARAM_5G.RSRQ,
        n_estimators: int = 100,
        random_state: int = 42,
        wknn_k: int = 2,
    ):
        self.n_clusters = n_clusters
        self.rf_param = rf_param
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.wknn_k = wknn_k

        # Values set in the fit method
        self.kmeans = None
        self.rf_model = None
        self.rps = None  # Reference points
        self.unique_pcis = None

        # Timing metrics
        self.training_time = None
        self.inference_time = None

    def fit(self, x):
        """
        Fit the localization model with the RP data,
        train clustering and classifier models
        :param x:
        :return:
        """
        start = perf_counter()

        # Set values
        self.rps = x
        self.unique_pcis = extract_unique_npcis(x["measurements_matrix"])

        if self.n_clusters == 0:
            # No clustering/classification
            self.training_time = perf_counter() - start
            return

        # Train Kmeans and RandomForest
        df_features = x[["lat", "lng"]].values
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(df_features)
        x["cluster"] = cluster_labels

        df_features, _ = create_point_matrix(self.rps, self.unique_pcis, self.rf_param)
        X = df_features
        y = self.rps["cluster"]

        rf_model = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        rf_model.fit(X, y)

        self.rf_model = rf_model
        self.training_time = perf_counter() - start

    def predict(self, x):
        start = perf_counter()
        if self.n_clusters == 0:
            # No clustering/classification: use all reference points for wKNN
            predicted_lat_lng = predict_lat_lng_wknn_no_cluster(
                df_tp=x,
                df_rp=self.rps,
                unique_npcis=self.unique_pcis,
                rf_param=self.rf_param,
                wknn_k=self.wknn_k,
            )
        else:
            # Cluster-based prediction
            predicted_lat_lng = predict_lat_lng_wknn(
                df_tp=x,
                df_rp=self.rps,
                unique_npcis=self.unique_pcis,
                rf_param=self.rf_param,
                wknn_k=self.wknn_k,
                rf_model=self.rf_model,
            )
        self.inference_time = perf_counter() - start
        return predicted_lat_lng

    def get_performance_stats(self, x, predicted_lat_lng, print_stats: bool = True):

        error = haversine_distance(
            x["lat"], x["lng"], predicted_lat_lng[:, 0], predicted_lat_lng[:, 1]
        )

        stats = {
            "median_error": error.median(),
            "mean_error": error.mean(),
            "std_error": error.std(),
            "min_error": error.min(),
            "max_error": error.max(),
            "training_time": self.training_time,
            "inference_time": self.inference_time,
        }

        if print_stats:
            print("==== Model Performance ====")
            table = [[k, f"{v:.2f}"] for k, v in stats.items()]
            print(tabulate(table, headers=["Stat", "Value"], tablefmt="github"))

        return error, stats


def predict_lat_lng_wknn(
    df_tp: pd.DataFrame,
    df_rp: pd.DataFrame,
    unique_npcis: np.array,
    rf_param: RF_PARAM_5G,
    wknn_k: int,
    rf_model,
):
    """
    Predict (lat, lng) for each test point using cluster-based wKNN.
    Returns: np.ndarray of shape (n_test_points, 2) with [lat, lng] in original order.
    """
    # Predict clusters for all test points
    tp_features, _ = create_point_matrix(df_tp, unique_npcis, rf_param)
    test_clusters = rf_model.predict(tp_features)
    df_tp = df_tp.copy()
    df_tp["predicted_cluster"] = test_clusters

    # Prepare result array
    predicted_positions = np.full((df_tp.shape[0], 2), np.nan)

    # Process each cluster
    for cluster in np.unique(test_clusters):
        idx = np.where(test_clusters == cluster)[0]
        tp_cluster = df_tp.iloc[idx]
        rp_cluster = df_rp[df_rp["cluster"] == cluster]
        if rp_cluster.empty:
            continue

        est_locs = process_points(
            tp_cluster, rp_cluster, unique_npcis, rf_param, wknn_k
        )

        # Assign estimated positions in the correct order
        predicted_positions[idx, :] = est_locs

    return predicted_positions


def predict_lat_lng_wknn_no_cluster(
    df_tp: pd.DataFrame,
    df_rp: pd.DataFrame,
    unique_npcis: np.array,
    rf_param: RF_PARAM_5G,
    wknn_k: int,
):
    """
    Predict (lat, lng) for each test point using wKNN with all reference points (no clustering).
    Returns: np.ndarray of shape (n_test_points, 2) with [lat, lng] in original order.
    """
    est_locs = process_points(df_tp, df_rp, unique_npcis, rf_param, wknn_k)
    return est_locs


def process_points(
    tps: pd.DataFrame,
    rps: pd.DataFrame,
    unique_npcis: np.array,
    rf_param: RF_PARAM_5G,
    wknn_k: int,
):
    # Create point matrices
    m_rp_full, idx_rp_full = create_point_matrix(rps, unique_npcis, rf_param)
    m_tp_full, idx_tp_full = create_point_matrix(tps, unique_npcis, rf_param)

    # Compute weights and sorted indices
    W, idx_sort = compute_weights(m_rp_full, idx_rp_full, m_tp_full, idx_tp_full)

    # Run wKNN
    est_locs, _ = wknn(tps, rps, idx_sort, W, k=wknn_k)
    return est_locs
