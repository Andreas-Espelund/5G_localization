import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from scripts.matrix_operations import create_point_matrix
from scripts.utils import RF_PARAM_5G, extract_unique_npcis


def train_kmeans(df: pd.DataFrame, n_clusters: int, random_state: int):
    """
    Do KMeans clustering based on lat,lng positions

    :param df: Reference points to cluster
    :param n_clusters: Number of clusters
    :param random_state: Random seed for reproducibility
    :return: KMeeans model and cluster labels
    """
    df_features = df[["lat", "lng"]].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(df_features)
    df["cluster"] = cluster_labels
    return kmeans, cluster_labels


def train_kmeans_rf_param(
    df: pd.DataFrame, n_clusters: int, rf_param: RF_PARAM_5G, random_state: int
):
    """
    Do KMeans clustering by radio parameter

    :param df: Reference points to cluster
    :param n_clusters: Number of clusters
    :param rf_param: RF parameter to cluster by
    :param random_state: Random seed for reproducibility
    :return: KMeeans model and cluster labels
    """
    pcis = extract_unique_npcis(df["measurements_matrix"])
    df_features, _ = create_point_matrix(df, pcis, rf_param)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(df_features)
    df["cluster"] = cluster_labels
    return kmeans, cluster_labels


def train_random_forest(
    df: pd.DataFrame,
    unique_npcis,
    rf_param: RF_PARAM_5G,
    n_estimators: int,
    random_state: int,
):
    """
    Train the random forest classifier.

    :param df: Reference points for model training
    :param unique_npcis: Set of unique npcis
    :param rf_param: RF param to train model on
    :param n_estimators: Number of estimators to train
    :param random_state: Random state for reproducibility
    :return: Random forest model
    """
    from scripts.weighted_coverage import create_point_matrix

    df_features, _ = create_point_matrix(df, unique_npcis, rf_param)

    X = df_features
    y = df["cluster"]

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    rf_model.fit(X, y)

    return rf_model
