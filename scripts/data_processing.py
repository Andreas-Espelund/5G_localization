import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from scripts.utils import RF_PARAM_5G


def cluster_data_and_train_random_forest(
    df: pd.DataFrame,
    df_rp: pd.DataFrame,
    n_clusters: int,
    unique_npcis,
    rf_param: RF_PARAM_5G,
    random_seed: int,
):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)

    #  === Cluster the data points using KMeans ===
    # if rf_param is None:
    features = df[["lat", "lng"]].values
    # else:
    #    features, _ = create_point_matrix(df, unique_npcis, rf_param)

    df["cluster"] = kmeans.fit_predict(features)

    #  === Train Random Forest Classifier ===

    rf_model = train_random_forest(
        df=df_rp,
        unique_npcis=unique_npcis,
        rf_param=rf_param,
        random_seed=random_seed * 42,
        n_estimators=100,
    )

    return rf_model


def cluster_data_and_train_kmeans_rf_param(
    df: pd.DataFrame,
    n_clusters: int,
    unique_npcis,
    rf_param: RF_PARAM_5G,
    random_seed: int,
):
    from scripts.matrix_operations import create_point_matrix

    df_features, _ = create_point_matrix(df, unique_npcis, rf_param)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)

    df["cluster"] = kmeans.fit_predict(df_features)

    rf_model = train_random_forest(
        df=df,
        unique_npcis=unique_npcis,
        rf_param=rf_param,
        random_seed=random_seed * 42,
        n_estimators=100,
    )

    return rf_model


def train_random_forest(
    df: pd.DataFrame,
    unique_npcis,
    rf_param: RF_PARAM_5G,
    random_seed: int,
    n_estimators: int = 100,
):
    from scripts.matrix_operations import create_point_matrix

    df_features, _ = create_point_matrix(df, unique_npcis, rf_param)

    X = df_features
    y = df["cluster"]

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_seed * 42
    )
    rf_model.fit(X, y)

    return rf_model
