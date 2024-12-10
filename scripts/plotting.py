import folium
import matplotlib.pyplot as plt
import pandas as pd

from scripts.utils import cluster_color_mapping

plt.rcParams.update(
    {
        "font.size": 32,  # Default text size
        "axes.titlesize": 32,  # Title size
        "axes.labelsize": 32,  # X and Y label size
        "xtick.labelsize": 32,  # X tick label size
        "ytick.labelsize": 32,  # Y tick label size
        "legend.fontsize": 26,  # Legend text size
        "figure.titlesize": 32,  # Figure title size
    }
)


def geo_plot_points(df: pd.DataFrame):
    """
    Plots given locations to a map (OpenStreetMap) that is viewable in broswer.
    Generates a file called 'map.html' in the current working directory.
    :param df:
    """
    # Create a map centered around the mean location
    m = folium.Map(location=[df["lat"].mean(), df["lng"].mean()], zoom_start=12)

    # Add CircleMarkers to the map
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lng"]],
            radius=5,  # Size of the marker
            color="blue",  # Border color of the marker
            fill=True,
            fill_color="blue",  # Fill color of the marker
            fill_opacity=0.6,
        ).add_to(m)

    # Save the map as an HTML file and open it in the browser
    m.save("map.html")


def make_boxplot(
    df: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    color: str = "forestgreen",
    baseline: float = None,
):
    """
    Creates a boxplot of the given dataframe.

    :param baseline:
    :param df:
    :param title:
    :param x_label:
    :param y_label:
    :param color: Color of the boxes
    """
    data_values = [df[col] for col in df.columns]

    plt.figure(figsize=(20, 12))
    median_props = dict(color="black", linewidth="3")
    plot = plt.boxplot(
        data_values,
        patch_artist=True,
        tick_labels=df.columns,
        medianprops=median_props,
    )
    for patch in plot["boxes"]:
        patch.set_facecolor(color)

    if baseline:
        plt.axhline(
            y=baseline,
            color="red",
            linestyle="-",
            linewidth=2,
            label="Baseline",
        )
        plt.legend(loc="upper right")

    # formatting the plot
    plt.title(title)
    plt.grid(axis="y")
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Show the plot
    plt.show()


def make_lat_lng_scatterplot(df: pd.DataFrame, col: str, col_label: str, title: str):
    plt.figure(figsize=(18, 14))

    for name, group in df.groupby(col):
        plt.scatter(
            group["lat"],
            group["lng"],
            c=cluster_color_mapping[name],
            label=f"{col_label} {name}",
            alpha=0.8,
        )
    offset = 0.00025
    # plot the points that have been placed in the wrong cluster
    for cluster, group in df.groupby("cluster"):
        misplaced = group[group["prediction"] != cluster]
        for _, row in misplaced.iterrows():
            plt.scatter(row["lat"], row["lng"], c="black", alpha=1, marker="d", s=250)

            # top triangle show the predicted cluster, shown with its color
            plt.scatter(
                row["lat"],
                row["lng"] + offset,
                c=cluster_color_mapping[row["prediction"]],
                alpha=1,
                marker="^",
                s=100,
            )
            # bottom triangle show the correct cluster, shown with its color
            plt.scatter(
                row["lat"],
                row["lng"] - offset,
                c=cluster_color_mapping[cluster],
                alpha=1,
                marker="v",
                s=100,
            )

    ncol = 1
    if df[col].nunique() > 9:
        ncol = 2
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.title(title)
    plt.legend(title=col_label, loc="lower left", ncol=ncol)
    plt.show()


def make_bar_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    color: str = "none",
    edgecolor: str = "forestgreen",
    hatch: str = "O",
    linewidth: int = 2,
    x_limits: (int, int) = None,
    y_limits: (int, int) = None,
    bar_labels: bool = False,
):
    """
    Creates a generic bar plot with customizable styling.

    :param bar_labels: Should draw values on the bars
    :param df: DataFrame containing the data to plot
    :param x_col: Column name for x-axis values
    :param y_col: Column name for y-axis values
    :param title: Title of the plot
    :param x_label: Label for the x-axis
    :param y_label: Label for the y-axis
    :param color: Fill color for bars
    :param edgecolor: Edge color for bars
    :param hatch: Hatch pattern for bars
    :param linewidth: Line width for bar edges
    :param x_limits: Tuple specifying x-axis limits
    :param y_limits: Tuple specifying y-axis limits
    """
    plt.figure(figsize=(20, 8))

    # Create a bar plot
    bars = plt.bar(
        df[x_col],
        df[y_col],
        color=color,
        edgecolor=edgecolor,
        hatch=hatch,
        linewidth=linewidth,
    )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(df[x_col])  # Ensure each bar is labeled with its x-axis value

    # Set x and y limits if specified
    if x_limits:
        plt.xlim(x_limits)
    if y_limits:
        plt.ylim(y_limits)

    # Format y-axis to display percentage if applicable
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}%"))

    # Annotate each bar with its height
    if bar_labels:
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                f"{yval:.1f}%",
                ha="center",
                va="bottom",
                fontsize="small",
            )

    plt.show()
