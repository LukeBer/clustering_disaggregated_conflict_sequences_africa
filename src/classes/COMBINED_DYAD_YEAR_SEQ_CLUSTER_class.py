import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.classes.DYAD_YEAR_SEQ_CLUSTER_class import DYAD_YEAR_SEQ_CLUSTER
from src.classes.SEQ_CLUSTER_class import SEQ_CLUSTER
from src.combine_sequences import (build_dataset_from_dyad_sequence,
                                   build_networks)
from src.data_loader import get_actor_dyad_mapping


class COMBINED_DYAD_YEAR_SEQ_CLUSTER(SEQ_CLUSTER):
    def __init__(
        self,
        df: pd.DataFrame,
        continuous: bool = False,
        MIN_SEQ_EVENTS: int = 10,  # For filtering sequences
        MIN_NUMBER_FATALITIES: int = 25,
        pre_seq_seq_mapping: pd.Series = pd.Series(),
        pre_event_seq_mapping: pd.Series = pd.Series(),
        pre_sequence_dict: dict = {},
        network=None,
        index: str = "month_year",
    ):
        """
        Initialize a combined dyad-year sequence clustering object.

        This class extends dyad-year sequences by combining them into larger sequences
        when they meet the following criteria:
        1. Dyad overlap – sequences share at least one actor.
        2. Temporal overlap – active periods overlap by ≥ 70%.
        3. Geographic proximity – mean event locations are within 200 km.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing conflict events, must include relevant dyad and event info.
        continuous : bool, optional
            Whether to treat dyads as continuous sequences (default: False).
        MIN_SEQ_EVENTS : int, optional
            Minimum number of events per combined sequence (default: 10).
        MIN_NUMBER_FATALITIES : int, optional
            Minimum fatalities per combined sequence (default: 25).
        pre_seq_seq_mapping : pd.Series, optional
            Pre-computed mapping of sequences for aggregation (default: empty Series).
        pre_event_seq_mapping : pd.Series, optional
            Pre-computed mapping of individual events to sequences (default: empty Series).
        pre_sequence_dict : dict, optional
            Pre-computed sequence dictionary for plotting and analysis (default: empty dict).
        network : networkx.Graph, optional
            Pre-computed network of dyad overlaps (default: None).
        index : str, optional
            Column to use as the temporal index for sequences (default: 'month_year').

        Attributes
        ----------
        event_seq_mapping : pd.Series
            Mapping of event IDs to their combined sequence labels.
        network : networkx.Graph
            Graph of combined sequences and their dyad overlaps.
        pre_seq_seq_mapping : pd.Series
            Mapping from combined sequence IDs to original dyad-year sequences.
        pre_sequence_dict : dict
            Dictionary of sequences with aggregated metrics for plotting.
        """
        super().__init__(
            "An extention of Actor-based Conflict Event Sequences, where sequences are combined in one larger sequences if all of these rules apply:\n1. Dyad overlap – their respective sequences have an overlap in their dyad attributes, meaning they have the same actor\n2. Temporal overlap – the active periods of the two sequences must overlap by at least a defined fraction (≥ 70% in our case).\n3. Geographic proximity – the mean event locations of the two sequences must be within a maximum distance threshold (200 km)."
        )
        self.continuous = continuous
        self.MIN_EVENT_NO = MIN_SEQ_EVENTS  # Map to base class property
        self.MIN_NO_FATALITIES = MIN_NUMBER_FATALITIES
        self.index = index
        df = df.copy()

        dyad_year_seq_cluster = DYAD_YEAR_SEQ_CLUSTER(
            df, MIN_SEQ_EVENTS=1, MIN_NUMBER_FATALITIES=0
        )
        self.pre_event_seq_mapping = dyad_year_seq_cluster.event_seq_mapping
        self.pre_sequence_dict = dyad_year_seq_cluster.extract_sequences(
            df=df, index=index
        )  # TODO index
        actor_dyad_mapping = get_actor_dyad_mapping(df)
        seq_data = build_dataset_from_dyad_sequence(dyad_year_seq_cluster)
        network = build_networks(seq_data, actor_dyad_mapping)
        self.network = network

        seq_data["combined_seq"] = ""
        components = list(nx.connected_components(network))
        for i, nodes in enumerate(components):
            seq_data.loc[seq_data["seq"].isin(nodes), "combined_seq"] = str(i)
        self.pre_seq_seq_mapping = (
            seq_data.groupby("combined_seq")["seq"].agg(list).copy()
        )
        seq_data = seq_data.drop(columns=["seq"])
        seq_data = seq_data.rename(columns={"combined_seq": "seq"})

        seq_data["total_fatalities"] = seq_data["fatalities"].apply(sum)
        seq_data["event_id_list"] = seq_data["event_id_list"].apply(
            lambda x: [item for sublist in x for item in sublist]
        )
        seq_data_grouped = (
            seq_data.groupby("seq")
            .agg(
                {
                    "event_id_list": lambda x: [i for sublist in x for i in sublist],
                    "total_fatalities": "sum",
                }
            )
            .reset_index()
        )
        seq_data_grouped.loc[
            seq_data_grouped["event_id_list"].apply(len) < self.MIN_EVENT_NO, "seq"
        ] = "no_cluster"
        seq_data_grouped.loc[
            seq_data_grouped["total_fatalities"] < self.MIN_NO_FATALITIES, "seq"
        ] = "no_cluster"
        seq_data_grouped = seq_data_grouped.drop(columns=["total_fatalities"])

        event_seq_mapper = seq_data_grouped[["event_id_list", "seq"]].explode(
            "event_id_list"
        )
        event_seq_mapper["event_id_cnty"] = event_seq_mapper["event_id_list"].astype(
            str
        )
        event_seq_mapper = event_seq_mapper.set_index("event_id_list")

        self.event_seq_mapping = event_seq_mapper["seq"]

    def extract_sequences(self, df):
        """
        Extract sequences using the base SEQ_CLUSTER extract_sequences method.

        Parameters
        ----------
        df : pd.DataFrame
            The event DataFrame to extract sequences from.

        Notes
        -----
        This method leverages `from_pre_sequence=True` to extract sequences from
        previously combined dyad-year sequences.
        """
        super().extract_sequences(df, self.index, from_pre_sequence=True)

    def plot_pre_sequences(
        self,
        num: int,
        fatalities_key: str = "fatalities",
        index_key: str = "index",
        grid_key: str = "gid",
        event_cnt_key: str = "event_id_cnty",
    ):
        """
        Plot aggregated pre-combined sequences for a given sequence ID.

        Parameters
        ----------
        num : int
            ID of the combined sequence to plot.
        fatalities_key : str, optional
            Column name for fatalities (default: 'fatalities').
        index_key : str, optional
            Column name for temporal index (default: 'index').
        grid_key : str, optional
            Column name for grid counts (default: 'gid').
        event_cnt_key : str, optional
            Column name for event counts (default: 'event_id_cnty').

        Notes
        -----
        Produces three plots:
        1. Events by sequence.
        2. Fatalities by sequence.
        3. Unique grid count by sequence.
        Each sub-sequence is color-coded by actor pair.
        """
        seq_list = self.pre_seq_seq_mapping.get(str(num))
        df = pd.DataFrame(self.pre_sequence_dict)

        if len(seq_list) == 1:
            super().plot_sequence(
                num, fatalities_key, index_key, grid_key, event_cnt_key
            )
            return

        # Prepare figure once
        _, ax = plt.subplots(figsize=(12, 6))
        _, ax_events = plt.subplots(figsize=(12, 6))
        _, ax_grid = plt.subplots(figsize=(12, 6))

        colors = plt.cm.get_cmap("tab20", len(seq_list))

        for i, seq in enumerate(seq_list):
            sub_df = df[df["seq"] == seq]

            if sub_df.empty:
                continue

            actor1 = sub_df["actor1"].values[0]
            actor2 = sub_df["actor2"].values[0]

            if isinstance(actor1, (list, tuple, np.ndarray)):
                actor1 = actor1[0]
            if isinstance(actor2, (list, tuple, np.ndarray)):
                actor2 = actor2[0]

            seq_label = f"{actor1} : {actor2}"

            x_vals = sub_df["index"].values[0]
            event_counts = sub_df[event_cnt_key].values[0]
            fatalities = sub_df[fatalities_key].values[0]
            grid_count = sub_df[grid_key].values[0]

            # --- Dynamically calculate bar width ---
            if isinstance(x_vals[0], (int, float, np.integer, np.floating)):
                # Use relative distance between x values
                diffs = np.diff(sorted(x_vals))
                bar_width = min(np.min(diffs), 1.0) * 0.8 if len(diffs) > 0 else 0.5
            else:
                # Fallback to fixed width for non-numeric x (e.g., datetime)
                bar_width = 5  # days or arbitrary small span

            ax_events.bar(
                x_vals,
                event_counts,
                color=colors(i),
                label=f"{seq_label}",
                width=bar_width,
            )
            ax_grid.bar(
                x_vals,
                grid_count,
                color=colors(i),
                label=f"{seq_label}",
                width=bar_width,
            )
            ax.plot(
                x_vals, fatalities, color=colors(i), marker="o", label=f"{seq_label}"
            )

        ax.set_xlabel(self.index_name)
        ax.set_ylabel("Count")
        ax.set_title(f"Aggregated Sequence {num}: Fatalities by Sequence")
        ax.legend()
        ax_grid.set_xlabel(self.index_name)
        ax_grid.set_ylabel("Count")
        ax_grid.set_title(f"Aggregated Sequence {num}: Unique Grid Count by Sequence")
        ax_grid.legend()
        ax_events.set_xlabel(self.index_name)
        ax_events.set_ylabel("Count")
        ax_events.set_title(f"Aggregated Sequence {num}: Events by Sequence")
        ax_events.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_network(self, num):
        """
        Plot the subnetwork corresponding to a combined sequence.

        Parameters
        ----------
        num : int
            Index of the combined sequence in the network components list.

        Notes
        -----
        Nodes are colored by `short_seq` attribute if available.
        Node labels show the actors of each sequence.
        Edges are drawn according to their 'style' attribute.
        The plot title shows number of nodes, edges, and total fatalities in the subnetwork.
        """
        components = list(nx.connected_components(self.network))
        nodes = components[num]
        # Your plotting code here using `nodes`

        subgraph = self.network.subgraph(nodes)

        total_fatalities = 0
        for n in subgraph.nodes:
            total_fatalities = total_fatalities + subgraph.nodes[n].get(
                "total_fatalities"
            )

        plt.figure(figsize=(6, 4))
        # pos = nx.spring_layout(subgraph, seed=42)
        pos = nx.kamada_kawai_layout(subgraph)
        # Node color by short_seq
        node_colors = [
            "orange" if subgraph.nodes[n].get("short_seq", False) else "skyblue"
            for n in subgraph.nodes
        ]

        # Draw nodes
        nx.draw_networkx_nodes(subgraph, pos=pos, node_color=node_colors, node_size=500)

        # Draw edges grouped by style
        edge_styles = nx.get_edge_attributes(subgraph, "style")

        for style in set(edge_styles.values()):
            styled_edges = [(u, v) for (u, v), s in edge_styles.items() if s == style]
            nx.draw_networkx_edges(
                subgraph, pos=pos, edgelist=styled_edges, style=style
            )

        node_labels = {
            n: f"{subgraph.nodes[n]['actors'][0]}\n{subgraph.nodes[n]['actors'][1]}"
            for n in subgraph.nodes
        }

        nx.draw_networkx_labels(
            subgraph, pos, labels=node_labels, font_size=6, verticalalignment="bottom"
        )

        title = f"Subnetwork {num}: {len(subgraph.nodes)} nodes, {subgraph.number_of_edges()} edges, {total_fatalities} total fatalities"
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
