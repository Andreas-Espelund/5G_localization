import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from scripts.utils import RF_PARAM, get_miss_ref_value


def create_point_matrix(
    df: pd.DataFrame, unique_npcis: np.array, rf_params: list[RF_PARAM]
):
    """
    Creates and populates a point matrix and valid index matrix for the test or reference points.
    The point matrix is populated with the given RF values or the miss_ref value.
    The idx matrix is populated with 1 or 0 based on if the point is valid.

    :param df: data points to create the matrix from
    :param unique_npcis: all the unique npcis to include in the point matrix
    :param rf_params: list of RF parameters to use
    :return: point matrix and idx matrix
    """
    num_points = df.shape[0]
    unique_npcis = [tuple(row) for row in unique_npcis]
    num_unique_npcis = len(unique_npcis)
    num_rf_params = len(rf_params)

    # Initialize the point matrix with an additional dimension for RF parameters
    point_matrix = np.zeros(
        shape=[num_points, num_unique_npcis, num_rf_params],
        dtype=np.float64,
    )

    # Fill with default values
    for idx, param in enumerate(rf_params):
        miss_value = get_miss_ref_value(param)
        point_matrix[:, :, idx] = miss_value

    idx_matrix = np.zeros(shape=[num_points, num_unique_npcis, num_rf_params])

    # Map each unique NPCI to its index
    npc_index_map = {npc: idx for idx, npc in enumerate(unique_npcis)}

    i = 0
    for _, row in df.iterrows():
        measurements = row["measurements_matrix"]
        for _, row in measurements.iterrows():
            npc_tuple = (row["NPCI"], row["eNodeBID"], row["operatorID"])
            if npc_tuple in npc_index_map:
                idx = npc_index_map[npc_tuple]
                for param_idx, rf_param in enumerate(rf_params):
                    rf_value = row[rf_param.value]
                    if not np.isnan(rf_value):
                        point_matrix[i, idx, param_idx] = rf_value
                        idx_matrix[i, idx, param_idx] = 1
        i += 1
    return point_matrix, idx_matrix


def compute_weights(
    m_rfp: np.array, idx_rfp: np.array, m_tp: np.array, idx_tp: np.array
) -> (np.array, np.array):
    """
    Computes weights for two matrices.
    This is used for the weights in wKNN.
    :param m_rfp: point matrix for the reference points
    :param idx_rfp: valid index matrix for the reference points
    :param m_tp: point matrix for the test points
    :param idx_tp: valid index matrix for the test points
    :return: Weights and sorted indices by weight
    """

    num_params = m_rfp.shape[2]

    # Compute the euclidian distances between the TPs and RPs for each RF Param
    D = np.zeros((m_tp.shape[0], m_rfp.shape[0], num_params))

    for param_idx in range(num_params):
        D[:, :, param_idx] = cdist(
            m_tp[:, :, param_idx], m_rfp[:, :, param_idx], metric="euclidean"
        )

    # Normalize distances based on common NPCIs
    match = np.logical_and(idx_tp[:, np.newaxis, :], idx_rfp[np.newaxis, :, :])

    s = np.sum(match, axis=2)

    # Avoid division by zero by setting distances to infinity where no matches exist
    D = np.divide(D, s, out=np.full_like(D, np.inf), where=s != 0)

    # Set distances to dummy reference points to a very large value
    dummy_rfps = np.all(idx_rfp == 0, axis=1)
    D[:, dummy_rfps] = np.inf

    # Replace zero distances with a small value to avoid singularities
    min_nonzero_distance = np.min(D[D > 0])
    D[D == 0] = min_nonzero_distance / 20

    # # turn D into a 2D array where the distances across different params are aggregated
    D = np.sum(D, axis=2)

    # Sort distances and compute weights
    idx_sort = np.argsort(D, axis=1)
    D_sort = np.take_along_axis(D, idx_sort, axis=1)
    W = 1.0 / D_sort

    return W, idx_sort
