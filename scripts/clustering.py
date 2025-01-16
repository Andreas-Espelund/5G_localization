import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from scripts.utils import RF_PARAM_5G


def train_kmeans(df: pd.DataFrame, n_clusters: int, random_state: int):
    coords = df[["lat", "lng"]].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(coords)
    return kmeans, cluster_labels


def train_random_forest(
    df: pd.DataFrame,
    unique_npcis,
    rf_param: RF_PARAM_5G,
    n_estimators: int,
    random_state: int,
):
    from scripts.weighted_coverage import create_point_matrix

    df_features, _ = create_point_matrix(df, unique_npcis, rf_param)

    X = df_features
    y = df["cluster"]

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    rf_model.fit(X, y)

    return rf_model


def train_cluster_classifier(
    df: pd.DataFrame,
    n_clusters: int,
    unique_npcis,
    rf_param: RF_PARAM_5G,
    random_seed: int,
):
    #  === Cluster the data points using KMeans ===
    _, cluster_labels = train_kmeans(df, n_clusters, random_seed)
    df["cluster"] = cluster_labels

    #  === Train Random Forest Classifier ===
    rf_model = train_random_forest(
        df=df,
        unique_npcis=unique_npcis,
        rf_param=rf_param,
        n_estimators=100,
        random_seed=random_seed,
    )

    return rf_model
