import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation, KMeans
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

from src.classes.SEQ_CLUSTER_class import SEQ_CLUSTER


def build_kmean_cluster(k, sequences: SEQ_CLUSTER, metric):
    """
    Apply KMeans clustering to sequence metrics.

    Parameters
    ----------
    k : int
        Number of clusters.
    sequences : SEQ_CLUSTER
        Sequence clustering object containing sequence_dict with metric arrays.
    metric : str
        Key in sequences.sequence_dict to cluster on.

    Returns
    -------
    SEQ_CLUSTER
        Original sequence object with updated 'cluster_label' in sequence_dict.

    Notes
    -----
    Prints the number of sequences in each cluster.
    """
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(sequences.sequence_dict[metric])

    sequences.sequence_dict["cluster_label"] = cluster_labels

    print(f"Clustering completed with {k} clusters")

    for i in range(k):
        count = sum(cluster_labels == i)
        percentage = count / len(cluster_labels) * 100
        print(f"  Cluster {i}: {count} sequences ({percentage:.1f}%)")
    return sequences


def build_numeric_feature_cluster(
    k, sequences: SEQ_CLUSTER, metric, method, reducer=np.nanmean, trim=0.05
):
    """
    Cluster sequences based on a numeric feature using binning.

    Parameters
    ----------
    k : int
        Number of clusters.
    sequences : SEQ_CLUSTER
        Sequence clustering object containing metric series.
    metric : str
        Key in sequence_dict containing numeric sequences.
    method : str
        Binning method, either 'quantile' or 'uniform'.
    reducer : callable, optional
        Function to reduce each sequence to a single numeric value (default: np.nanmean).
    trim : float, optional
        Fraction of extreme values to trim before binning (default: 0.05).

    Returns
    -------
    SEQ_CLUSTER
        Updated sequence object with 'cluster_label'.

    Notes
    -----
    Handles NaN values by imputing with the mean.
    """
    metric_series = sequences.sequence_dict[metric]
    if metric_series.empty:
        raise ValueError("No sequences available for clustering.")

    values = np.fromiter(
        (reducer(np.asarray(seq, dtype=float)) for seq in metric_series),
        dtype=float,
        count=len(metric_series),
    )
    if np.isnan(values).any():
        fill_value = np.nanmean(values)
        if np.isnan(fill_value):
            raise ValueError("Reducer returned only NaNs; cannot impute.")
        values = np.where(np.isnan(values), fill_value, values)
    if method == "quantile":
        edges = _metric_quantile_bins(values, k, trim)
    elif method == "uniform":
        edges = _metric_bins(values, k, trim)
    intervals = list(zip(edges[:-1], edges[1:]))

    cluster_labels = (
        np.zeros_like(values, dtype=int)
        if np.all(edges == edges[0])
        else np.digitize(values, edges[1:-1], right=True)
    )
    sequences.sequence_dict["cluster_label"] = cluster_labels

    print(f"Binning completed with {k} clusters")
    total = len(cluster_labels)
    for i, (lo, hi) in enumerate(intervals):
        count = int(np.sum(cluster_labels == i))
        pct = (count / total) * 100 if total else 0.0
        print(
            f"  Cluster {i}: {count} sequences ({pct:.1f}%) range [{lo:.2f}, {hi:.2f}]"
        )

    return sequences

def build_affinity_propagation_cluster(
    sequences: SEQ_CLUSTER,
    metric: str,
    preference: float = None,
    damping: float = 0.95,
    convergence_iter: int = 50,
    max_iter: int = 500,
    random_state: int = 1
):
    
    X = np.asarray(sequences.sequence_dict[metric], dtype=float)

    ap_clusterer = AffinityPropagation(
        random_state=random_state,
        damping=damping,
        preference=preference,
        convergence_iter=convergence_iter,
        max_iter=max_iter
    )

    labels = ap_clusterer.fit_predict(X)
    sequences.sequence_dict["cluster_label"] = labels

    unique_labels = np.unique(labels)
    print(f"AffinityPropagation discovered {len(unique_labels)} clusters")
    total = len(labels)
    for lab in unique_labels:
        count = int(np.sum(labels == lab))
        pct = (count / total) * 100.0
        print(f"  Cluster {lab}: {count} sequences ({pct:.1f}%)")
    return sequences

def _metric_bins(values, k: int, trim: float):
    """
    Generate uniform bins for numeric values with optional trimming.

    Parameters
    ----------
    values : array-like
        Numeric values.
    k : int
        Number of bins.
    trim : float
        Fraction to trim from each tail.

    Returns
    -------
    np.ndarray
        Array of bin edges.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("Metric series is empty.")
    if np.isnan(array).all():
        raise ValueError("All metric values are NaN; cannot form bins.")

    true_min = np.nanmin(array)
    true_max = np.nanmax(array)
    if true_min == true_max:
        return np.full(k + 1, true_min, dtype=float)

    if trim > 0.0:
        lower = np.nanpercentile(array, trim * 100.0)
        upper = np.nanpercentile(array, (1.0 - trim) * 100.0)
        array = np.clip(array, lower, upper)

    edges = np.linspace(np.nanmin(array), np.nanmax(array), k + 1, dtype=float)
    edges[0] = true_min
    edges[-1] = true_max
    return edges


def _metric_quantile_bins(values, k: int, trim: float = 0.0):
    """
    Generate quantile-based bins for numeric values with optional trimming.

    Parameters
    ----------
    values : array-like
        Numeric values.
    k : int
        Number of bins.
    trim : float
        Fraction to trim from tails (default: 0).

    Returns
    -------
    np.ndarray
        Array of bin edges.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("Metric series is empty.")
    if np.isnan(array).all():
        raise ValueError("All metric values are NaN; cannot form bins.")
    if k < 1:
        raise ValueError("k must be at least 1.")

    # Remove infinities, keep NaNs for later imputation
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("No finite values available for binning.")

    if trim > 0.0:
        lb = np.nanpercentile(finite, trim * 100.0)
        ub = np.nanpercentile(finite, (1.0 - trim) * 100.0)
        finite = finite[(finite >= lb) & (finite <= ub)]
        if finite.size == 0:
            raise ValueError("Trimmed range removed all finite values.")

    quantiles = np.linspace(0.0, 1.0, k + 1)
    edges = np.quantile(finite, quantiles, method="linear")

    # Preserve full range so outliers still land in first/last bin
    true_min = np.nanmin(array)
    true_max = np.nanmax(array)
    edges[0] = true_min
    edges[-1] = true_max
    return edges


def plot_cluster(sequences: SEQ_CLUSTER, metric):
    """
    Plot sequences grouped by cluster with mean curve overlay.

    Parameters
    ----------
    sequences : SEQ_CLUSTER
        Sequence object containing 'cluster_label' and metric arrays.
    metric : str
        Key in sequence_dict for plotting.

    Notes
    -----
    - Uses log scale on x-axis.
    - Mean sequence per cluster is shown in red.
    - Handles variable-length sequences with NaN padding.
    """
    normalized_df = pd.DataFrame(sequences.sequence_dict)
    actual_clusters = sorted(normalized_df["cluster_label"].unique())
    n_clusters = len(actual_clusters)

    # Grid setup
    n_cols = 4
    n_rows = (n_clusters + n_cols - 1) // n_cols
    _, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True, sharey=True
    )
    axes = axes.flatten()

    for idx, cluster in enumerate(actual_clusters):
        ax = axes[idx]
        cluster_data = normalized_df[normalized_df["cluster_label"] == cluster]

        # Convert sequences to arrays
        curves = [np.array(seq) for seq in cluster_data[metric]]

        # Find max length *for this cluster only*
        max_len = max(len(c) for c in curves)

        # Plot each curve with its own x-range
        for curve in curves:
            ax.plot(range(len(curve)), curve, alpha=0.5, linewidth=1)

        # Compute mean curve with NaN padding
        padded_curves = np.full((len(curves), max_len), np.nan)
        for i, curve in enumerate(curves):
            padded_curves[i, : len(curve)] = curve

        mean_curve = np.nanmean(padded_curves, axis=0)
        ax.plot(range(max_len), mean_curve, color="red", linewidth=2.5, label="Mean")

        # Adjust axes to fit this cluster's length
        ax.set_xscale("log")
        ax.set_xlim(1, max_len)

        # Titles, labels, grid, etc.
        n_sequences = len(cluster_data)
        ax.set_title(f"Cluster {cluster} ({n_sequences} seq.)", fontsize=9)
        ax.grid(True, alpha=0.2)

        if idx % n_cols == 0:
            ax.set_ylabel(metric, fontsize=8)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("Time Bin", fontsize=8)

        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.legend(fontsize=7, loc="upper left")

    for j in range(len(actual_clusters), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_cluster_summary(cluster, index, metric):
    """
    Produce a multi-panel cluster summary with environmental, locational, event, duration, and fatalities plots.

    Parameters
    ----------
    cluster : SEQ_CLUSTER
        Sequence clustering object containing sequences and cluster labels.
    index : str
        Key representing the time index of sequences.
    metric : str
        Metric used for plotting curves.
    """
    env_cols = [
        "agri_gc",
        "aquaveg_gc",
        "barren_gc",
        "forest_gc",
        "herb_gc",
        "shrub_gc",
        "urban_gc",
        "water_gc",
    ]
    loc_cols = ["ttime_mean", "dist_border_km_2004", "dist_capital_km_2004"]

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], polar=True)
    ax3 = fig.add_subplot(gs[0, 2], polar=True)
    ax4 = fig.add_subplot(gs[1, 0])

    # one axis for events and duration combined
    ax5_events = fig.add_subplot(gs[1, 1])  # big axis
    ax6 = fig.add_subplot(gs[1, 2])

    # your existing plots
    plot_sequence_curve(cluster, index, metric, ax=ax1)
    plot_spider_chart(env_cols, cluster, "Environment by Cluster", ax=ax2)
    plot_spider_chart(loc_cols, cluster, "Location by Cluster", ax=ax3)
    plot_relative_disorder_type_by_cluster(cluster, ax=ax4)

    # --- overlay boxplots
    # draw events (left y-axis)
    plot_cluster_boxplots(cluster, "event_id_cnty", "", "", ax=ax5_events)
    ax5_events.set_ylabel("Number of Events")
    ax5_events.set_title("Events per Sequence by Cluster")

    # create twin axis for duration
    ax5_duration = ax5_events.twinx()
    plot_cluster_duration_summary(cluster, index, ax=ax5_duration)

    # fatalities stays as is
    plot_cluster_boxplots(
        cluster,
        "fatalities",
        "Number of Fatalities per Sequence by Cluster",
        "Number of Fatalities",
        ax=ax6,
    )

    plt.tight_layout()
    plt.show()


def plot_sequence_curve(cluster, index, metric, ax):
    """
    Plot mean sequence curve per cluster with standard deviation shading.

    Parameters
    ----------
    cluster : SEQ_CLUSTER
        Sequence object containing sequences and cluster labels.
    index : str
        Time index column in sequence_dict.
    metric : str
        Metric to plot.
    ax : matplotlib.axes.Axes
        Axis to plot on.

    Returns
    -------
    matplotlib.axes.Axes
        Axis with plotted curves.
    """
    df = pd.DataFrame(cluster.sequence_dict)
    # Explode only the list columns
    df_exploded = df.explode([index, metric])

    # Make sure numeric
    df_exploded[index] = pd.to_numeric(df_exploded[index])
    df_exploded[metric] = pd.to_numeric(df_exploded[metric])

    # Group by cluster and binned_index
    grouped = df_exploded.groupby(["cluster_label", index])

    # Compute mean and std
    stats = grouped[metric].agg(["mean", "std"]).reset_index()

    for cluster_label in stats["cluster_label"].unique():
        cluster_data = stats[stats["cluster_label"] == cluster_label]
        x = cluster_data[index]
        y = cluster_data["mean"]
        yerr = cluster_data["std"]

        ax.plot(x, y, label=f"Cluster {cluster_label}")
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)

    ax.set_xlabel("Index")
    ax.set_ylabel(metric)
    ax.set_title("Mean Sequence Curve per Cluster with Variance Band")
    ax.legend()
    return ax


def plot_spider_chart(env_cols, cluster, title, ax):
    """
    Plot radar/spider chart of cluster-level mean features.

    Parameters
    ----------
    env_cols : list of str
        List of feature columns to plot.
    cluster : SEQ_CLUSTER
        Sequence object containing sequences and cluster labels.
    title : str
        Title of the chart.
    ax : matplotlib.axes.Axes
        Axis to plot on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    df = pd.DataFrame(cluster.sequence_dict)

    # Group by cluster
    for col in env_cols:
        df[col] = df[col].apply(np.mean)

    clusters = df.groupby("cluster_label")[env_cols].mean()

    # Number of variables
    categories = env_cols
    N = len(categories)

    # Compute angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # close the circle

    # Plot each cluster
    for cluster, row in clusters.iterrows():
        values = row.tolist()
        values += values[:1]  # close the polygon
        ax.plot(
            angles, values, linewidth=2, linestyle="solid", label=f"Cluster {cluster}"
        )
        ax.fill(angles, values, alpha=0.25)

    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title, size=14, weight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    return ax


def plot_cluster_boxplots(cluster, col, title, y_label, ax):
    """
    Plot boxplots of a column aggregated per cluster.

    Parameters
    ----------
    cluster : SEQ_CLUSTER
        Sequence object with cluster labels.
    col : str
        Column to summarize.
    title : str
        Plot title.
    y_label : str
        Y-axis label.
    ax : matplotlib.axes.Axes
        Axis to plot on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    df = pd.DataFrame(cluster.sequence_dict)

    clusters = sorted(df["cluster_label"].unique())

    df[col] = df[col].apply(sum)
    positions = np.arange(len(clusters))

    # Prepare boxplot data
    plot_data = [df.loc[df["cluster_label"] == cl, col] for cl in clusters]

    bp = ax.boxplot(
        plot_data,
        positions=positions,
        widths=0.6,
        labels=[str(cl) for cl in clusters],
        showfliers=False,
        patch_artist=False,
    )

    # keep everything visible and consistent with the twin axis
    ax.set_xlim(-0.5, len(clusters) - 0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(cl) for cl in clusters])

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", alpha=0.3)

    return ax


def plot_relative_disorder_type_by_cluster(cluster, ax):
    """
    Plot relative frequency of disorder types per cluster.

    Parameters
    ----------
    cluster : SEQ_CLUSTER
        Sequence object containing 'disorder_type' lists and cluster labels.
    ax : matplotlib.axes.Axes
        Axis to plot on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    df = pd.DataFrame(cluster.sequence_dict)

    # Flatten nested disorder lists
    df["disorder_type"] = df["disorder_type"].apply(
        lambda x: [i for sub in x for i in (sub if isinstance(sub, list) else [sub])]
    )

    # Explode to long format
    df_exploded = df[["disorder_type", "cluster_label"]].explode("disorder_type")

    # Count occurrences
    disorder_counts = (
        df_exploded.groupby(["cluster_label", "disorder_type"])
        .size()
        .reset_index()
        .rename(columns={0: "frequency"})
    )

    # Compute relative frequencies per cluster
    disorder_counts["relative_freq"] = 0
    for cluster_label in disorder_counts["cluster_label"].unique():
        cluster_data = disorder_counts[
            disorder_counts["cluster_label"] == cluster_label
        ]
        disorder_counts.loc[
            disorder_counts["cluster_label"] == cluster_label, "relative_freq"
        ] = (cluster_data["frequency"] / cluster_data["frequency"].sum())

    # Pivot for plotting
    plot_data = disorder_counts.pivot(
        index="disorder_type", columns="cluster_label", values="relative_freq"
    ).fillna(0)

    # --- Plot grouped bars ---
    x = np.arange(len(plot_data.index))  # positions for disorder types
    width = 0.8 / len(plot_data.columns)  # bar width split among clusters

    for i, cluster_label in enumerate(plot_data.columns):
        ax.bar(
            x + i * width,
            plot_data[cluster_label],
            width,
            label=f"Cluster {cluster_label}",
        )

    ax.set_xticks(x + width * (len(plot_data.columns) - 1) / 2)
    ax.set_xticklabels(plot_data.index, rotation=45, ha="right")
    ax.set_ylabel("Relative Frequency")
    ax.set_xlabel("Disorder Type")
    ax.set_title("Relative Disorder Types by Cluster")
    ax.legend(title="Cluster")

    return ax


def _sequence_duration_days(index_sequence):
    """
    Compute duration in days from a sequence of datetime-like indices.

    Parameters
    ----------
    index_sequence : list-like
        Sequence of timestamps or datetime-like objects.

    Returns
    -------
    float
        Duration in days, NaN if invalid sequence.
    """
    if index_sequence is None or len(index_sequence) == 0:
        return np.nan

    arr = pd.to_datetime(np.asarray(index_sequence, dtype=object), errors="coerce")
    mask = ~arr.isna()
    if not mask.any():
        return np.nan

    start = arr[mask][0]  # arrays are sorted chronologically
    end = arr[mask][-1]
    return (end - start).days + 1


def plot_cluster_duration_summary(cluster, index_key: str, ax):
    """
    Annotate cluster x-axis with mean sequence duration.

    Parameters
    ----------
    cluster : SEQ_CLUSTER
        Sequence object containing time index sequences and cluster labels.
    index_key : str
        Key for time index sequences.
    ax : matplotlib.axes.Axes
        Axis to annotate with mean duration per cluster.

    Notes
    -----
    - Displays mean duration under each cluster tick.
    - Adds a single explanatory line below all ticks.
    """
    df = pd.DataFrame(cluster.sequence_dict)
    df["duration_days"] = df["index"].apply(_sequence_duration_days)

    # mean per cluster
    g = df.groupby("cluster_label")["duration_days"].mean()
    order = np.sort(df["cluster_label"].unique())
    means = g.reindex(order)

    # clean axis
    ax.clear()
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # cluster ticks
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=0)
    ax.tick_params(axis="x", pad=26)  # more space below ticks

    # annotate means directly under each tick (smaller font + more offset)
    for x, m in enumerate(means):
        txt = "–" if pd.isna(m) else f"{int(round(m))}"
        ax.annotate(
            txt,
            xy=(x, 0),
            xycoords=ax.get_xaxis_transform(),
            xytext=(0, -17),  # move further below
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            rotation=45,
        )

    # a single explanatory line below all ticks
    ax.text(
        0.5,
        -0.17,
        "Duration per Sequence by Cluster",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
