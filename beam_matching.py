from scripts.data_writer import save_experiment_result


def main():
    import numpy as np

    from scripts.data_filter import filter_dataframe
    from scripts.data_loader import load_dataframe
    from scripts.utils import RF_PARAM_5G, NETWORK_TYPE

    filename = "5G_data_2023.mat"

    # Series of random seeds for reproducability
    random_seeds = np.loadtxt("data/random_seeds.csv", dtype=int)

    # load the dataframe from saved file or 'raw' matlab file
    df = load_dataframe(filename, NETWORK_TYPE._5G)

    # # Drop unused columns to save space
    matrix_cols_to_drop = ["toa_pps", "toa_cir", "toa_cov", "campaign_id"]
    df["measurements_matrix"] = df["measurements_matrix"].apply(
        lambda x: x.drop(columns=matrix_cols_to_drop)
    )

    operator_choice = [1, 10]
    selected_campaigns = list(range(1, 11))
    rf_param = RF_PARAM_5G.RSRQ

    # Data filtering
    df = filter_dataframe(
        df=df,
        operators=operator_choice,
        include_columns=[
            "pci",
            "beam_index",
            "nr_arfcn",
            "operator_id",
            rf_param.value,
        ],
        campaigns=selected_campaigns,
    )

    from scripts.matrix_operations import create_point_matrix
    from scripts.single import create_point_vector, compute_weights_single, wknn_single
    from scripts.utils import extract_unique_npcis, dataset_tp_rp_split
    from scripts.beamforming import get_best_beam, find_matching_rps
    import pandas as pd

    def test_point(tp: pd.Series, df_rp: pd.DataFrame, use_beam_matching=True):
        matches = tp["matches"]

        if use_beam_matching:
            rps = df_rp.loc[matches]
        else:
            rps = df_rp

        if rps.empty:
            print("\rNo matches found")
            raise Exception("No matches found")

        # Create point matrices/vectors
        m_tp, idx_tp = create_point_vector(tp, unique_npcis, rf_param)
        m_rp, idx_rp = create_point_matrix(rps, unique_npcis, rf_param)

        # Compute weights
        W, idx_sort = compute_weights_single(m_rp, idx_rp, m_tp, idx_tp)

        # Estimate location
        location_est, error = wknn_single(tp, rps, idx_sort, W, 2)
        return error, rps.shape[0]

    # iterate over tps
    errors = []
    unique_npcis = extract_unique_npcis(df["measurements_matrix"])

    df = df.sample(1000)
    df["best_beam"] = df["measurements_matrix"].apply(
        lambda x: get_best_beam(x, rf_param)
    )

    df_tp, df_rp = dataset_tp_rp_split(df, 0.3, 42)

    df_tp["matches"] = df_tp["best_beam"].apply(lambda x: find_matching_rps(df_rp, x))

    total = df_tp.shape[0]

    from concurrent.futures import ThreadPoolExecutor

    def process_row(args):
        i, row, df_rp = args
        try:
            error, length = test_point(row, df_rp, True)
            error_control, control_length = test_point(row, df_rp, False)
            print(f"\r {i}/ {total}", end="")
            return (i, error, error_control, length, control_length)
        except Exception as e:
            print(f"\r Error processing row {i}: {str(e)}", end="")
            return None

    # Create arguments list
    args_list = [(i, row, df_rp) for i, row in df_tp.iterrows()]

    # Process with 20 threads
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(process_row, args_list))

    # Filter out None results and sort
    errors = [result for result in results if result is not None]
    errors = sorted(errors, key=lambda x: x[0])

    results_df = pd.DataFrame(
        errors, columns=["i", "error", "error_control", "length", "control_length"]
    )

    config = {}

    data = {
        "errors": results_df,
    }

    save_experiment_result("beam_matching_experiment_new", config, data)
