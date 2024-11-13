import folium
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

plt.rcParams.update({
    'font.size': 22,  # Default text size
    'axes.titlesize': 26,  # Title size
    'axes.labelsize': 26,  # X and Y label size
    'xtick.labelsize': 22,  # X tick label size
    'ytick.labelsize': 22,  # Y tick label size
    'legend.fontsize': 20,  # Legend text size
    'figure.titlesize': 30  # Figure title size
})


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


def make_boxplot(df: pd.DataFrame, title: str, x_label: str, y_label: str):
    """
    Creates a boxplot of the given dataframe.

    :param df:
    :param title:
    :param x_label:
    :param y_label:
    """
    data_values = [df[col] for col in df.columns]

    plt.figure(figsize=(20, 12))
    median_props = dict(color="black", linewidth="3")
    plot = plt.boxplot(
        data_values,
        patch_artist=True,
        labels=df.columns,
        medianprops=median_props
    )
    hatch_pattern = 'O'
    hatch_color = 'forestgreen'

    for patch in plot['boxes']:
        patch.set(hatch=hatch_pattern, edgecolor=hatch_color)
        patch.set_facecolor('none')
        patch.set_edgecolor(hatch_color)

    # formatting the plot
    plt.title(title)
    plt.grid(axis="y")
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Show the plot
    plt.show()


def make_lat_lng_scatterplot(df: pd.DataFrame, col: str, col_label: str, title: str, plot_individual: bool = False):
    campaign_color_mapping = {
        0: 'orange',
        1: 'red',
        2: 'green',
        3: 'blue',
        4: 'yellow',
        5: 'purple',
        6: 'teal'
    }

    cluster_color_mapping = {
        0: 'orange',
        1: 'red',
        2: 'green',
        3: 'blue',
        4: 'yellow',
        5: 'purple',
        6: 'teal',
        7: 'pink',
        8: 'brown',
        9: 'gray',
        10: 'cyan',
        11: 'magenta',
        12: 'lime',
        13: 'navy',
        14: 'maroon',
        15: 'olive',
        16: 'silver',
        17: 'gold',
        18: 'lavender',
        19: 'wheat',
        20: 'turquoise'
    }

    if col == 'campaign_id':
        map = campaign_color_mapping

    if col == 'prediction':
        map = cluster_color_mapping

    if map is None:
        raise ValueError('Invalid map')

    plt.figure(figsize=(8, 6))
    for name, df in df.groupby(col):
        plt.scatter(df['lat'], df['lng'], c=map[name], label=f"{col_label} {name}", alpha=0.2)

    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title(title)
    plt.legend(title=col_label)
    plt.show()

    if not plot_individual: return

    for name, df in df.groupby(col):
        plt.figure(figsize=(5, 3))
        plt.scatter(df['lat'], df['lng'], c=map[name], label=f"{col_label} {name}", alpha=1)
        plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        plt.title(f"{title} for {col_label} {name}")
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.legend(title=col_label, bbox_to_anchor=(-0.1, -0.1))
        plt.show()


def make_bar_plot(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str,
        x_label: str,
        y_label: str,
        color: str = 'none',
        edgecolor: str = 'forestgreen',
        hatch: str = 'O',
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
    plt.figure(figsize=(10, 8))

    # Create a bar plot
    bars = plt.bar(
        df[x_col],
        df[y_col],
        color=color,
        edgecolor=edgecolor,
        hatch=hatch,
        linewidth=linewidth
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
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))

    # Annotate each bar with its height
    if bar_labels:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.1f}%', ha='center', va='bottom',
                     fontsize='small')

    plt.show()
