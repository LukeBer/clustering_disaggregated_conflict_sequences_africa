import datetime
from itertools import chain
from typing import List

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.constants import DEFAULT_METRIC_AGG_MAP
from src.utils.normalization import (convert_index_to_relative,
                                     normalize_seq_cluster)


class SEQ_CLUSTER:

    def __init__(self, description: str):
        self.description = description
        self.event_seq_mapping: pd.Series = pd.Series()
        # sequence_dict will be a dictionary with keys like 'index', 'event_id_cnty', 'fatalities', etc., and values are arrays of arrays (one per cluster)
        self.sequence_dict: dict = {}
        self.MIN_EVENT_NO: int = 10
        self.MIN_NO_FATALITIES: int = 25
        self.metric_mapping: dict
        self.index_name: str
        """
        Initialize a SEQ_CLUSTER object.

        Parameters
        ----------
        description : str
            A textual description of the sequence clustering object or analysis context.

        Attributes
        ----------
        event_seq_mapping : pd.Series
            Series mapping event IDs to sequence IDs.
        sequence_dict : dict
            Dictionary storing extracted sequences and associated metrics.
        MIN_EVENT_NO : int
            Minimum number of events for a sequence to be considered valid.
        MIN_NO_FATALITIES : int
            Minimum number of fatalities for a sequence to be considered.
        metric_mapping : dict
            Mapping of metric names to aggregation functions.
        index_name : str
            Name of the index used in sequence aggregation (e.g., 'year', 'event_date').
        """

    def extract_sequences(
        self,
        df: pd.DataFrame,
        index: str,
        agg_dict: dict = DEFAULT_METRIC_AGG_MAP,
        from_pre_sequence: bool = False,
    ) -> dict:
        """
        Extract sequences from a DataFrame of conflict events, aggregating metrics by sequence and index.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing event-level conflict data.
        index : str
            Column name to use as the temporal index for aggregation (e.g., 'year', 'event_date').
        agg_dict : dict, optional
            Dictionary mapping metric columns to aggregation types (default: DEFAULT_METRIC_AGG_MAP).
        from_pre_sequence : bool, optional
            Whether to build sequences from a previously defined sequence mapping (default: False).

        Returns
        -------
        dict
            Dictionary of sequences with aggregated metrics.
        """

        df["event_id_list"] = df["event_id_cnty"].copy()
        agg_dict = agg_dict.copy()
        
        if not all(key in df.columns for key in agg_dict.keys()):
            missing_keys = [key for key in agg_dict.keys() if key not in df.columns]
            raise ValueError(
                f"The following aggregation keys are missing from the DataFrame: {missing_keys}"
            )

        if from_pre_sequence:
            agg = {"pre_seq": "list"}
            agg_dict.update(agg)

            df["pre_seq"] = df["event_id_cnty"].map(self.pre_event_seq_mapping)

        df["seq"] = df["event_id_cnty"].map(self.event_seq_mapping)
        df = df[df["seq"] != "no_cluster"]

        self.metric_mapping = agg_dict
        self.index_name = index
        agg_dict = self._map_metric_agg_dict(agg_dict)

        # First group: by ['seq', index]
        grouped = df.groupby(["seq", index]).agg(agg_dict).reset_index()

        # collect lists of index and metrics per seq
        final_agg = {
            index: list,
            "event_id_cnty": list,
        }
        for metric in agg_dict:
            final_agg[metric] = list

        aggregated_df = grouped.groupby("seq").agg(final_agg).reset_index()

        # Rename index column to 'index'
        aggregated_df = aggregated_df.rename(columns={index: "index"})

        # Build sequence_dict directly, converting list values to arrays
        self.sequence_dict = {}
        for col in aggregated_df.columns:
            # TODO be dynamic
            if (
                col == "seq"
                or col == "event_id_list"
                or col == "pre_seq"
                or col == "disorder_type"
            ):
                self.sequence_dict[col] = aggregated_df[col].to_numpy()
            else:
                # This gives an array of arrays
                self.sequence_dict[col] = aggregated_df[col].apply(np.array)

        return self.sequence_dict

    def _map_metric_agg_dict(self, metric_map: dict) -> dict:
        """
        Convert human-readable aggregation types to pandas-compatible aggregation functions.

        Parameters
        ----------
        metric_map : dict
            Dictionary mapping metric names to aggregation type strings (e.g., 'sum', 'count', 'mean').

        Returns
        -------
        dict
            Dictionary suitable for use in pandas .agg() calls.
        """
        agg_func_map = {}

        for col, agg in metric_map.items():
            if agg == "count":
                agg_func_map[col] = "count"
            elif agg == "sum":
                agg_func_map[col] = "sum"
            elif agg == "mean":
                agg_func_map[col] = "mean"
            elif agg == "first":
                agg_func_map[col] = "first"
            elif agg == "list":
                agg_func_map[col] = lambda x: (
                    [item for sublist in x for item in sublist]
                    if all(isinstance(i, list) for i in x)
                    else list(x)
                )
            elif agg == "unique count":
                agg_func_map[col] = pd.Series.nunique
            elif agg == "length":
                agg_func_map[col] = lambda x: (
                    x.str.len().sum() if x.dtype == "O" else len(x)
                )
            else:
                raise ValueError(
                    f"Unsupported aggregation type: {agg} for column '{col}'"
                )

        return agg_func_map

    def apply_normalization(self, normalization_config: dict):
        """
        Normalize sequence metrics using a user-defined normalization configuration.

        Parameters
        ----------
        normalization_config : dict
            Dictionary specifying normalization operations for each metric (e.g., proportional, integral).

        Returns
        -------
        SEQ_CLUSTER
            The SEQ_CLUSTER object with normalized metrics.
        """
        return normalize_seq_cluster(self, normalization_config)

    def describe_sequences(self):
        """
        Print an overview of the extracted sequences, including:
        - Total sequences
        - Number of events in and out of sequences
        - Sequence length statistics (mean, std, min, max)
        - Sequence duration statistics
        - Summary statistics for each aggregated metric
        """

        if not hasattr(self, "sequence_dict") or not self.sequence_dict:
            print("No sequences extracted yet.")
            return

        no_cluster_count = len(
            self.event_seq_mapping[self.event_seq_mapping.values == "no_cluster"]
        )
        total_events = len(self.event_seq_mapping)
        in_sequence = total_events - no_cluster_count

        print(f"{'='*40}")
        print("Sequence Overview")
        print(f"{'='*40}")
        print(f"{'Total sequences:':35} {len(self.sequence_dict['index'])}")
        print(f"{'Events in a sequence:':35} {in_sequence}")
        print(
            f"{'Events not in a sequence:':35} {no_cluster_count} (min events: {self.MIN_EVENT_NO})\n"
        )

        # Sequence length
        n_events = self.sequence_dict["event_id_cnty"].apply(sum)
        print(f"{'-'*40}")
        print("Sequence Length (Number of Events)")
        print(f"{'-'*40}")
        print(f"{'Mean':10} {np.mean(n_events):.2f}")
        print(f"{'Std':10} {np.std(n_events):.2f}")
        print(f"{'Min':10} {min(n_events)}")
        print(f"{'Max':10} {max(n_events)}\n")

        # Sequence duration
        time_spans = []
        for index_vals in self.sequence_dict["index"]:
            first_val = index_vals[0]
            if isinstance(first_val, (pd.Timestamp, datetime.datetime, datetime.date)):
                delta = max(index_vals) - min(index_vals)
                time_spans.append(delta.days / 365.25)
            elif isinstance(first_val, (int, float, np.number)):
                time_spans.append(max(index_vals) - min(index_vals))
            else:
                time_spans.append(None)

        if all(ts is not None for ts in time_spans):
            print(f"{'-'*40}")
            print(
                f"Sequence Duration (Index: {self.index_name})"
            )  # TODO:is it index or year
            print(f"{'-'*40}")
            print(f"{'Mean':10} {np.mean(time_spans):.2f}")
            print(f"{'Std':10} {np.std(time_spans):.2f}")
            print(f"{'Min':10} {min(time_spans):.2f}")
            print(f"{'Max':10} {max(time_spans):.2f}\n")

        # Metric summaries
        print(f"{'-'*40}")
        print("Metric Distribution Summary")
        print(f"{'-'*40}")
        for metric, agg_func in self.metric_mapping.items():
            if agg_func in ["count", "sum", "mean", "unique count", "length"]:
                values = self.sequence_dict[metric].apply(np.mean)
                print(f"{metric} ({agg_func}):")
                print(f"  {'Mean':8} {np.mean(values):.4f}")
                print(f"  {'Std':8} {np.std(values):.4f}")
                print(f"  {'Min':8} {np.min(values):.4f}")
                print(f"  {'Max':8} {np.max(values):.4f}\n")

    def plot_sequence(
        self,
        num: int,
        fatalities_key: str = "fatalities",
        index_key: str = "index",
        grid_key: str = "gid",
        event_cnt_key: str = "event_id_cnty",
    ):
        """
        Plot a single sequence showing the number of events, fatalities, and spatial spread (grids) over time.

        Parameters
        ----------
        num : int
            Index of the sequence to plot.
        fatalities_key : str, optional
            Column name in sequence_dict for fatalities (default: 'fatalities').
        index_key : str, optional
            Column name in sequence_dict for the temporal index (default: 'index').
        grid_key : str, optional
            Column name in sequence_dict for the number of unique grids (default: 'gid').
        event_cnt_key : str, optional
            Column name in sequence_dict for the number of events (default: 'event_id_cnty').
        """

        if not hasattr(self, "sequence_dict") or not self.sequence_dict:
            print("No sequences extracted yet.")
            return

        required_keys = [index_key, fatalities_key, event_cnt_key, grid_key]
        for key in required_keys:
            if key not in self.sequence_dict:
                raise ValueError(f"Key '{key}' not found in sequence_dict.")

        x = self.sequence_dict[index_key][num]
        event_counts = self.sequence_dict[event_cnt_key][num]
        fatalities = self.sequence_dict[fatalities_key][num]
        gid_counts = self.sequence_dict[grid_key][num]

        # --- Dynamically calculate bar width ---
        if isinstance(x[0], (int, float, np.integer, np.floating)):
            # Use relative distance between x values
            diffs = np.diff(sorted(x))
            bar_width = min(np.min(diffs), 1.0) * 0.8 if len(diffs) > 0 else 0.5
        else:
            # Fallback to fixed width for non-numeric x (e.g., datetime)
            bar_width = 5  # days or arbitrary small span

        # --- Plot 1: Events and Fatalities ---
        _, ax = plt.subplots(figsize=(12, 4))
        ax.bar(x, event_counts, color="skyblue", label="Event Count", width=bar_width)
        ax.plot(x, fatalities, color="red", marker="o", label="Fatalities")
        ax.set_xlabel(self.index_name)
        ax.set_ylabel("Count")
        ax.set_title(
            f"Sequence {self.sequence_dict['seq'][num]}: Events and Fatalities"
        )
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # --- Plot 2: GID counts ---
        plt.figure(figsize=(12, 4))
        plt.bar(x, gid_counts, color="purple", width=bar_width)
        plt.xlabel(self.index_name)
        plt.ylabel("Unique GRID Count")
        plt.title(
            f"Sequence {self.sequence_dict['seq'][num]}: Number of grids in which sequence was active"
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_sequence_metrics_matrix(self, metric: str):
        """
        Visualize metrics across all sequences in a 2x2 matrix:
        - Histogram of number of events per sequence
        - Histogram of average metric values per sequence
        - Scatter plot of duration vs number of events
        - Scatter plot of duration vs average metric value

        Parameters
        ----------
        metric : str
            Metric name to plot (must be present in metric_mapping).
        """

        if not hasattr(self, "sequence_dict") or not self.sequence_dict:
            print("No sequences extracted yet.")
            return
        if not metric in self.metric_mapping.keys():
            print("Metric not available.")
            return

        # Collect metrics
        index_vals = self.sequence_dict["index"]
        metric_vals = self.sequence_dict[metric]

        # Calculate metrics
        n_events = self.sequence_dict["event_id_cnty"].apply(sum)
        avg_metric = self.sequence_dict[metric].apply(
            lambda x: np.mean(x) if isinstance(x, np.ndarray) else metric_vals
        )

        # Duration calculation
        relative_index = self.sequence_dict["index"].apply(
            lambda x: convert_index_to_relative(x, self.index_name)
        )
        duration = relative_index.apply(lambda x: max(x) - min(x))

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Sequence Metrics Overview", fontsize=16)

        # Plot 1: Number of events distribution
        axes[0, 0].hist(n_events, bins=20, alpha=0.7, edgecolor="black")
        axes[0, 0].set_title("Distribution of Event Counts")
        axes[0, 0].set_xlabel("Number of Events")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Average metric distribution
        axes[0, 1].hist(avg_metric, bins=20, alpha=0.7, edgecolor="black")
        axes[0, 1].set_title(f"Distribution of Average {metric} over {self.index_name}")
        axes[0, 1].set_xlabel(f"Average {metric}")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Duration vs Events
        axes[1, 0].scatter(duration, n_events, alpha=0.6)
        axes[1, 0].set_title("Number of Events vs Duration")
        axes[1, 0].set_xlabel(f"Duration ({self.index_name})")
        axes[1, 0].set_ylabel("Number of Events")
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Average metric vs Events
        axes[1, 1].scatter(duration, avg_metric, alpha=0.6)
        axes[1, 1].set_title(f"Average {metric} vs Duration")
        axes[1, 1].set_xlabel("Duration")
        axes[1, 1].set_ylabel(f"Average {metric}")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_metric_boxplot(self, metric: str):
        """
        Plot boxplots for a given metric across all sequences, with and without outliers.

        Parameters
        ----------
        metric : str
            Metric name to visualize (must be present in metric_mapping).
        """
        if not hasattr(self, "sequence_dict") or not self.sequence_dict:
            print("No sequences extracted yet.")
            return
        if metric not in self.metric_mapping.keys():
            print("Metric not available.")
            return

        # Flatten and collect all metric values from all sequences
        metric_values = np.concatenate(self.sequence_dict[metric].values)

        if metric_values.size == 0:
            print("No metric values found to plot.")
            return

        _, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

        sns.boxplot(y=metric_values, ax=axes[0], showfliers=True)
        axes[0].set_title(f"{metric} (with outliers)")
        axes[0].set_ylabel(metric)
        axes[0].set_xticks([])

        sns.boxplot(y=metric_values, ax=axes[1], showfliers=False)
        axes[1].set_title(f"{metric} (no outliers)")
        axes[1].set_ylabel(metric)
        axes[1].set_xticks([])

        plt.suptitle(
            f"Distribution of Aggregated Metric Values over Sequences: {metric}",
            fontsize=14,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def plot_map(self, num, df):
        """
        Plot the geographic trajectory of a sequence on a map, showing event locations, fatalities,
        and connecting events chronologically. Optionally labels dyads if sequence has fewer than 10 events.

        Parameters
        ----------
        num : int
            Index of the sequence to plot.
        df : pd.DataFrame
            DataFrame containing event-level data with latitude, longitude, fatalities, actor names, and event IDs.
        """
        flat_unique = set(chain.from_iterable(self.sequence_dict["event_id_list"][num]))
        df = df[df["event_id_list"].isin(flat_unique)]

        # Ensure 'event_date' is datetime
        df["event_date"] = pd.to_datetime(df["event_date"])
        df_sorted = df.sort_values("event_date")

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df_sorted,
            geometry=gpd.points_from_xy(df_sorted["longitude"], df_sorted["latitude"]),
            crs="EPSG:4326",  # WGS84 lat/lon
        )

        gdf = gdf.to_crs(epsg=3857)
        _, ax = plt.subplots(figsize=(10, 8))

        # Plot points sized by fatalities
        gdf.plot(ax=ax, markersize=gdf["fatalities"] * 10, color="red", alpha=0.5)

        # Connect points in chronological order
        ax.plot(gdf.geometry.x, gdf.geometry.y, color="gray", linestyle="--", alpha=0.3)

        # Add labels for dyads only if fewer than 10 events
        if len(gdf) < 10:
            for x, y, label in zip(
                gdf.geometry.x, gdf.geometry.y, gdf["actor1"] + " \n " + gdf["actor2"]
            ):
                ax.text(x + 1000, y + 1000, label, fontsize=5)

        ctx.add_basemap(ax, source="OpenStreetMap.Mapnik")
        ax.set_axis_off()
        plt.title("Conflict Events by Dyad on Map")
        plt.show()
