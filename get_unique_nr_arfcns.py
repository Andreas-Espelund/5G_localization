import pandas as pd

from scripts.data_loader import load_dataframe
from scripts.utils import NETWORK_TYPE


def find_unique_nr_arfcns():
    filename = "5G_data_2023.mat"

    # load the dataframe from saved file or 'raw' matlab file
    df = load_dataframe(filename, NETWORK_TYPE._5G)

    # # Drop unused columns to save space
    matrix_cols_to_drop = ["toa_pps", "toa_cir", "toa_cov", "campaign_id"]
    df["measurements_matrix"] = df["measurements_matrix"].apply(
        lambda x: x.drop(columns=matrix_cols_to_drop)
    )

    print("loaded dataframe")

    result_df = pd.DataFrame()

    for _, row in df.iterrows():
        measurements_matrix = row["measurements_matrix"]

        res = measurements_matrix[["nr_arfcn", "operator_id"]].drop_duplicates()
        result_df = pd.concat([result_df, res], ignore_index=True)

    result_df.drop_duplicates(inplace=True)

    print("processed dataframe")

    result_df.to_csv("nr_arfcn_operators_no_freq.csv", index=False)

    print("saved dataframe")

    print(result_df)

    """
    the nr_arfcn found here are calculated to frequency
    using this online tool: https://5g-tools.com/5g-nr-arfcn-calculator/
    """


def print_freq_overview():

    df = pd.read_csv("nr_arfcn_operators.csv")

    for name, group in df.groupby("operator_id"):
        print(name)
        print(group["frequency"].tolist())


def main():
    print_freq_overview()


if __name__ == "__main__":
    main()
