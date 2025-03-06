import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from scripts.utils import (
    get_miss_ref_value,
    RF_PARAM_5G,
    create_df_index_map,
    apply_index_map,
)


def create_point_matrix(
    df: pd.DataFrame, unique_npcis: np.array, rf_param: RF_PARAM_5G, best_beam=None
):
    """
    Creates and populates a point matrix and valid index matrix for the test or reference points.
    The point matrix is populated with the given RF value or the miss_ref value.
    The idx matrix is populated with 1 or 0 based on if the point is valid.

    :param df: data points to create the matrix from
    :param unique_npcis: all the unique npcis to include in the point matrix
    :param rf_param: RF parameter to use
    :return: point matrix and idx matrix
    """
    num_points = df.shape[0]
    unique_npcis = [tuple(row) for row in unique_npcis]
    num_unique_npcis = len(unique_npcis)

    # Initialize the point matrix and index matrix
    point_matrix = np.zeros(
        shape=[num_points, num_unique_npcis],
        dtype=np.float64,
    )

    # Fill with default values
    miss_value = get_miss_ref_value(rf_param)
    point_matrix[:, :] = miss_value

    idx_matrix = np.zeros(shape=[num_points, num_unique_npcis])

    # Map each unique NPCI to its index
    npc_index_map = {npc: idx for idx, npc in enumerate(unique_npcis)}

    i = 0
    for _, row in df.iterrows():
        measurements = row["measurements_matrix"]
        for _, measurement_row in measurements.iterrows():
            npc_tuple = (
                measurement_row["pci"],
                measurement_row["beam_index"],
                measurement_row["nr_arfcn"],
                measurement_row["operator_id"],
            )

            if npc_tuple in npc_index_map:
                idx = npc_index_map[npc_tuple]
                rf_value = measurement_row[rf_param.value]
                if not np.isnan(rf_value):
                    point_matrix[i, idx] = rf_value
                    idx_matrix[i, idx] = 1
        i += 1

    return point_matrix, idx_matrix


def compute_weights(
    m_rfp: np.array,
    idx_rfp: np.array,
    m_tp: np.array,
    idx_tp: np.array,
    df_tp: pd.DataFrame = None,
    df_rp: pd.DataFrame = None,
) -> (np.array, np.array):
    """
    Computes weights for two matrices with a single reference point parameter.
    :param df_tp:
    :param m_rfp: point matrix for the reference points (2D array)
    :param idx_rfp: valid index matrix for the reference points (1D array)
    :param m_tp: point matrix for the test points (2D array)
    :param idx_tp: valid index matrix for the test points (1D array)
    :return: Weights and sorted indices by weight
    """

    # Compute the Euclidean distances between the TPs and RPs
    D = cdist(m_tp, m_rfp, metric="euclidean")

    # Normalize distances based on common valid indices
    match = np.logical_and(idx_tp[:, np.newaxis, :], idx_rfp[np.newaxis, :, :])
    s = np.sum(match, axis=2)

    # Avoid division by zero by setting distances to a very large value where no matches exist
    realmax = np.finfo(np.float64).max
    D = np.divide(D, s, out=np.full_like(D, realmax), where=s != 0)

    # Set distances to dummy reference points to a very large value
    dummy_rfps = np.all(idx_rfp == 0, axis=1)
    D[:, dummy_rfps] = realmax

    # Replace zero distances with a small value to avoid singularities
    min_nonzero_distance = np.min(D[D > 0])
    D[D == 0] = min_nonzero_distance / 20

    if df_tp is not None and df_rp is not None:
        mapping = create_df_index_map(df_rp)
        for i in range(df_tp.shape[0]):
            row = df_tp.iloc[i, :]
            matches = row["matches"].tolist()
            matches = apply_index_map(matches, mapping)
            non_matching_indices = set(range(m_rfp.shape[0])) - set(matches)
            D[i, list(non_matching_indices)] = realmax

            if i == 1:
                print("matches", matches)

    # Sort distances and compute weights
    idx_sort = np.argsort(D, axis=1)
    D_sort = np.take_along_axis(D, idx_sort, axis=1)
    W = 1.0 / D_sort

    return W, idx_sort
