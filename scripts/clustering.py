import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from scripts.matrix_operations import create_point_matrix
from scripts.utils import RF_PARAM_5G, extract_unique_npcis


def train_dbscan(features: pd.DataFrame, eps: float, min_samples: int):
    """
    Train DBSCAN clustering model using RF parameter features

    Args:
        features: DataFrame with RF parameter features
        eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other
        min_samples: The number of samples in a neighborhood for a point to be considered as a core point
    """
    # Standardize the features as RF parameters might be on different scales
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(features_scaled)
    return dbscan, cluster_labels, scaler


# def train_kmeans(df: pd.DataFrame, n_clusters: int, random_state: int):
#     coords = df[["lat", "lng"]].values
#     kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
#     cluster_labels = kmeans.fit_predict(coords)
#     return kmeans, cluster_labels


def train_kmeans(
    df: pd.DataFrame, rf_param: RF_PARAM_5G, n_clusters: int, random_state: int
):
    unique_pcis = extract_unique_npcis(df["measurements_matrix"])
    df_features, _ = create_point_matrix(
        df,
        unique_npcis=unique_pcis,
        rf_param=rf_param,
    )
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(df_features)
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


def train_cluster_classifier_dbscan(
    df: pd.DataFrame,
    unique_npcis,
    rf_param: RF_PARAM_5G,
    eps: float = 0.5,  # Adjusted default since we're using standardized features
    min_samples: int = 5,
    random_seed: int = 42,
):
    from scripts.weighted_coverage import create_point_matrix

    # === Create feature matrix using RF parameters ===
    df_features, _ = create_point_matrix(df, unique_npcis, rf_param)

    # === Cluster the data points using DBSCAN on RF parameter features ===
    _, cluster_labels, _ = train_dbscan(
        features=df_features, eps=eps, min_samples=min_samples
    )
    df["cluster"] = cluster_labels

    # === Train Random Forest Classifier ===
    rf_model = train_random_forest(
        df=df,
        features=df_features,
        n_estimators=100,
        random_state=random_seed,
    )

    return rf_model, cluster_labels
