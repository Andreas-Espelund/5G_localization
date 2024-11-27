import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from scripts.utils import RF_PARAM, get_miss_ref_value


def train_kmeans(df_rp: pd.DataFrame, n_clusters: int, random_state: int):
    """
    Train a k-means model on the reference points.
    :param df_rp: DataFrame of reference points
    :param n_clusters: Number of clusters
    :param random_state: Random state for reproducibility
    :return: Trained k-means model and cluster labels for the reference points
    """
    coords = df_rp[['lat', 'lng']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(coords)
    return kmeans, cluster_labels


def aggregate_nsinr_by_npci(measurements_matrix, rf_param: RF_PARAM):
    # Group by NPCI and calculate the mean NSINR for each group
    return measurements_matrix.groupby('NPCI')[rf_param.value].mean().to_dict()


def flatten_matrix_values_to_columns(df: pd.DataFrame, rf_param: RF_PARAM):
    df['rf_feature'] = df['measurements_matrix'].apply(lambda x: aggregate_nsinr_by_npci(x, rf_param))
    npcis_df = df['rf_feature'].apply(pd.Series)
    df = pd.concat([df, npcis_df], axis=1)
    df.drop(columns=['rf_feature'], inplace=True)

    feature_columns = npcis_df.columns
    default_value = get_miss_ref_value(rf_param)
    df[feature_columns].fillna(default_value)

    return df[feature_columns]


def train_random_forest(df: pd.DataFrame, unique_npcis, rf_param: RF_PARAM, n_estimators: int, random_state: int):
    from scripts.weighted_coverage import create_point_matrix
    df_features, _ = create_point_matrix(df, unique_npcis, rf_param)

    X = df_features
    y = df['cluster']

    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X, y)

    return rf_model
