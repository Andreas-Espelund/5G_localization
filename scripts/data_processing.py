import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from scripts.utils import RF_PARAM
from scripts.weighted_coverage import create_point_matrix


def cluster_data_and_train_random_forest(
        df: pd.DataFrame,
        n_clusters: int,
        unique_npcis,
        rf_param: RF_PARAM,
        random_seed: int
):
    #  === Cluster the data points using KMeans ===

    coords = df[['lat', 'lng']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)
    df['cluster'] = kmeans.fit_predict(coords)

    #  === Train Random Forest Classifier ===

    # Prepare params and features
    n_estimators_rf = 100
    df_features, _ = create_point_matrix(df, unique_npcis, rf_param)
    X = df_features
    y = df['cluster']

    # Train and fit the model
    rf_model = RandomForestClassifier(n_estimators=n_estimators_rf, random_state=random_seed * 42)
    rf_model.fit(X, y)

    return rf_model
