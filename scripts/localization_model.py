import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from scripts.matrix_operations import create_point_matrix, compute_weights
from scripts.utils import RF_PARAM_5G, extract_unique_npcis
from scripts.weighted_coverage import wknn_one


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

    def fit(self, x):
        """
        Fit the localization model with the RP data,
        train clustering and classifier models
        :param x:
        :return:
        """
        df_features = x[["lat", "lng"]].values
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(df_features)
        x["cluster"] = cluster_labels

        self.rps = x
        self.unique_pcis = extract_unique_npcis(x["measurements_matrix"])

        df_features, _ = create_point_matrix(self.rps, self.unique_pcis, self.rf_param)

        X = df_features
        y = self.rps["cluster"]

        rf_model = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        rf_model.fit(X, y)

        self.rf_model = rf_model

    def predict(self, x):
        predicted_lat_lng = predict_lat_lng_wknn(
            df_tp=x,
            df_rp=self.rps,
            unique_npcis=self.unique_pcis,
            rf_param=self.rf_param,
            wknn_k=self.wknn_k,
            rf_model=self.rf_model,
        )

        return predicted_lat_lng


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

        # Create point matrices
        m_rp_full, idx_rp_full = create_point_matrix(rp_cluster, unique_npcis, rf_param)
        m_tp_full, idx_tp_full = create_point_matrix(tp_cluster, unique_npcis, rf_param)

        # Compute weights and sorted indices
        W, idx_sort = compute_weights(m_rp_full, idx_rp_full, m_tp_full, idx_tp_full)

        # Run wKNN
        est_locs, _ = wknn_one(tp_cluster, rp_cluster, idx_sort, W, k=wknn_k)

        # Assign estimated positions in the correct order
        predicted_positions[idx, :] = est_locs

    return predicted_positions
