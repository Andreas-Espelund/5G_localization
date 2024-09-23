import numpy as np

RSSI, NSINR, NRSRP, NRSRQ = "RSSI", "NSINR", "NRSRP", "NRSRQ"
MISS_REF_VALUES = {RSSI: -160, NSINR: -40, NRSRQ: -40, NRSRP: -160}


def get_miss_ref_value(rf_param: str) -> int:
    return MISS_REF_VALUES[rf_param]


def create_point_matrix(df, unique_npcis, rf_param):
    num_points = df.shape[0]
    num_unique_npcis = len(unique_npcis)
    miss_ref_value = get_miss_ref_value(rf_param)

    point_matrix = np.full(
        shape=[num_points, num_unique_npcis], fill_value=miss_ref_value
    )

    idx_matrix = np.zeros(shape=[num_points, num_unique_npcis])

    for i in range(num_points):
        measurements = df.iloc[i]["measurements_matrix"]
        for _, row in measurements.iterrows():
            npc = row["NPCI"]
            rf_value = row[rf_param]
            if npc in unique_npcis:
                idx = np.where(unique_npcis == npc)[0][0]
                point_matrix[i, idx] = (
                    rf_value if not np.isnan(rf_value) else miss_ref_value
                )
                idx_matrix[i, idx] = 1 if not np.isnan(rf_value) else 0

    return point_matrix, idx_matrix
