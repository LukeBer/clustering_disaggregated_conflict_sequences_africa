from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from src.constants import MAX_DIST_KM, MAX_SECOND_DEGREE_DIST, THR_TIMEOVERLAP
from src.utils.geographic import haversine


def build_dataset_from_dyad_sequence(dyad_year_seq_cluster):
    """
    Construct a DataFrame from a DYAD sequence cluster object.

    Parameters
    ----------
    dyad_year_seq_cluster : object
        An instance of a sequence cluster (e.g., DYAD_YEAR_SEQ_CLUSTER) containing
        a `sequence_dict` with sequences, indices, lat/lon, fatalities, etc.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - short_seq: Boolean indicating whether a sequence is short.
        - median_latitude / median_longitude: Median coordinates of the sequence.
        - dyad: Dyad identifier.
        - time_start / time_end: Start and end timestamps of the sequence.
    """
    sequence_data = pd.DataFrame(dyad_year_seq_cluster.sequence_dict)
    sequence_data["short_seq"] = indentify_short_sequences(
        dyad_year_seq_cluster, sequence_data
    )
    sequence_data["median_latitude"] = sequence_data["latitude"].apply(
        lambda arr: np.median(arr)
    )
    sequence_data["median_longitude"] = sequence_data["longitude"].apply(
        lambda arr: np.median(arr)
    )
    split_seq = sequence_data["seq"].str.split(" | ")
    sequence_data["dyad"] = split_seq.str[0].str.strip()

    sequence_data["time_start"] = sequence_data["index"].apply(min)
    sequence_data["time_end"] = sequence_data["index"].apply(max)
    return sequence_data


def indentify_short_sequences(dyad_year_seq_cluster, sequence_data):
    """
    Identify short sequences based on median events and fatalities.

    Parameters
    ----------
    dyad_year_seq_cluster : object
        Sequence cluster containing `sequence_dict`.
    sequence_data : pd.DataFrame
        DataFrame constructed from the sequence cluster.

    Returns
    -------
    pd.Series
        Boolean series where True indicates the sequence is short.
    """
    total_events = sequence_data["event_id_cnty"].apply(sum)
    total_fatalities = sequence_data["fatalities"].apply(sum)
    len_distribution = dyad_year_seq_cluster.sequence_dict["event_id_cnty"].apply(sum)
    median_no_events = len_distribution.quantile(0.5)
    fatal_distribution = dyad_year_seq_cluster.sequence_dict["fatalities"].apply(sum)
    median_fatalities = fatal_distribution.quantile(0.5)
    return ~((total_events > median_no_events) & (total_fatalities > median_fatalities))


def build_networks(sequence_data, actor_to_dyads_df):
    """
    Construct a dyad network graph from sequence data.

    Parameters
    ----------
    sequence_data : pd.DataFrame
        DataFrame containing sequences, dyads, coordinates, start/end times, and short_seq flags.
    actor_to_dyads_df : pd.DataFrame
        DataFrame mapping actors to dyads for connecting sequences.

    Returns
    -------
    nx.Graph
        NetworkX graph with nodes representing sequences and edges representing
        temporal overlap and geographic proximity.

    Notes
    -----
    - Removes second-degree edges that are too far.
    - Removes bridges to prevent weak links in the network.
    - Downsizes subnetworks with excessive edges.
    - Prints summary information about nodes and small sequences.
    """
    G = nx.Graph()
    for _, row in sequence_data.iterrows():
        seq = row["seq"]
        G.add_node(
            seq,
            short_seq=row["short_seq"],
            actors=[row["actor1"][0], row["actor2"][0]],
            lon=row["median_longitude"],
            lat=row["median_latitude"],
            total_fatalities=sum(row["fatalities"]),
        )

    for _, dyad_list in zip(actor_to_dyads_df["actor"], actor_to_dyads_df["dyad"]):
        dyad_info = sequence_data.loc[sequence_data["dyad"].isin(dyad_list)].set_index(
            "seq"
        )[
            [
                "time_start",
                "time_end",
                "short_seq",
                "median_longitude",
                "median_latitude",
            ]
        ]

        for s1, s2 in combinations(dyad_info.index, 2):

            if check_time_overlap(dyad_info, s1, s2):
                if check_distance(dyad_info, s1, s2):
                    G.add_edge(s1, s2, style="solid")

    G = remove_second_degree_connections_to_far(G)
    G = remove_bridges(G)
    G = downsize(G)

    print_info(G)
    return G


def check_time_overlap(dyad_info, s1, s2):
    """
    Determine if two sequences overlap in time sufficiently to add an edge.

    Parameters
    ----------
    dyad_info : pd.DataFrame
        DataFrame containing start/end times, short_seq flags, and median coordinates.
    s1, s2 : str
        Sequence identifiers.

    Returns
    -------
    bool
        True if sequences meet temporal overlap criteria.

    Notes
    -----
    - Long↔short sequences only connect if the short sequence starts after the long one.
    - Long↔long or short↔short sequences must exceed a threshold fraction overlap (THR_TIMEOVERLAP).
    """
    t1s, t1e = dyad_info.loc[s1, ["time_start", "time_end"]]
    t2s, t2e = dyad_info.loc[s2, ["time_start", "time_end"]]

    # compute overlap duration
    ov_start = max(t1s, t2s)
    ov_end = min(t1e, t2e)
    ov_dur = max(1, (ov_end - ov_start).days)

    is_short1, is_short2 = (
        dyad_info.loc[s1, "short_seq"],
        dyad_info.loc[s2, "short_seq"],
    )
    is_long1, is_long2 = not is_short1, not is_short2

    add_edge = False

    # long ↔ short
    if is_long1 ^ is_short2:
        if is_short1 and is_long2:
            t_short_start = t1s
            t_long_start = t2s
        else:
            t_short_start = t2s
            t_long_start = t1s

        # Only add edge if short started after long
        if t_short_start >= t_long_start:
            add_edge = True

    # long ↔ long or short ↔ short
    else:
        dur1 = (t1e - t1s).days
        dur2 = (t2e - t2s).days
        frac1 = ov_dur / dur1 if dur1 > 0 else 0
        frac2 = ov_dur / dur2 if dur2 > 0 else 0
        if frac1 >= THR_TIMEOVERLAP and frac2 >= THR_TIMEOVERLAP:
            add_edge = True
    return add_edge


def check_distance(dyad_info, s1, s2):
    """
    Determine if two sequences are geographically close enough to connect.

    Parameters
    ----------
    dyad_info : pd.DataFrame
        DataFrame with median_latitude and median_longitude for sequences.
    s1, s2 : str
        Sequence identifiers.

    Returns
    -------
    bool
        True if distance is within MAX_DIST_KM.
    """
    lat1, lon1 = (
        dyad_info.at[s1, "median_latitude"],
        dyad_info.at[s1, "median_longitude"],
    )
    lat2, lon2 = (
        dyad_info.at[s2, "median_latitude"],
        dyad_info.at[s2, "median_longitude"],
    )
    dist_km = haversine(lat1, lon1, lat2, lon2)
    return dist_km <= MAX_DIST_KM


def remove_second_degree_connections_to_far(G):
    """
    Remove edges to second-degree neighbors that are too far apart.

    Parameters
    ----------
    G : nx.Graph
        Input network graph.

    Returns
    -------
    nx.Graph
        Graph with second-degree edges exceeding MAX_SECOND_DEGREE_DIST removed.

    Notes
    -----
    - Uses haversine distance between node coordinates.
    - Only removes offending edges, retains closer neighbors.
    """
    MAX_SECOND_DEGREE_DIST = 300  # km

    edges_to_drop = []
    for n, d in G.nodes(data=True):
        # Get all second-degree neighbors (neighbors of neighbors)
        second_deg_neighbors = set()
        for nb in G[n]:
            second_deg_neighbors.update(G[nb])
        second_deg_neighbors.discard(n)  # remove self

        # Check distance to all second-degree neighbors
        too_far = False
        for sdn in second_deg_neighbors:
            lat1, lon1 = d["lat"], d["lon"]
            lat2, lon2 = G.nodes[sdn].get("lat"), G.nodes[sdn].get("lon")
            dist_km = haversine(lat1, lon1, lat2, lon2)
            if dist_km > MAX_SECOND_DEGREE_DIST:
                too_far = True
                break

        if too_far:
            edges_to_drop.append((n, nb))
    G.remove_edges_from(edges_to_drop)
    return G


def remove_bridges(G):
    """
    Remove bridge edges from the graph to avoid weak links between subnetworks.

    Parameters
    ----------
    G : nx.Graph
        Input network graph.

    Returns
    -------
    nx.Graph
        Graph with all bridge edges removed.

    Notes
    -----
    - Only considers connected components with more than 2 nodes.
    """
    larger_than_two_subnetworks = [
        G.subgraph(c).copy() for c in nx.connected_components(G) if len(c) > 2
    ]

    all_bridges = []
    for subgraph in larger_than_two_subnetworks:
        bridges = list(nx.bridges(subgraph))
        all_bridges.extend(bridges)

    G.remove_edges_from(all_bridges)
    return G


def print_info(G):
    """
    Print summary information about the graph.

    Parameters
    ----------
    G : nx.Graph
        Input network graph.

    Notes
    -----
    Prints:
    - Total nodes
    - Total edges
    - Total small sequences in connected subnetworks with edges
    """
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")

    total_small_sequences = sum(
        sum(subgraph.nodes[n].get("short_seq", False) for n in subgraph.nodes)
        for c in nx.connected_components(G)
        for subgraph in [G.subgraph(c)]
        if subgraph.number_of_edges() > 0
    )

    print(f"Total small sequences in subnetwork: {total_small_sequences}")


def downsize(G):
    """
    Reduce the number of edges in dense subnetworks by removing highly connected nodes.

    Parameters
    ----------
    G : nx.Graph
        Input network graph.

    Returns
    -------
    nx.Graph
        Downsized graph where subnetworks with >50 edges are iteratively trimmed.

    Notes
    -----
    - Removes the highest-degree node edges iteratively until edge count <= 50.
    - Operates on connected components separately.
    """
    for c in list(nx.connected_components(G)):
        subgraph = G.subgraph(c).copy()  # work on a copy to identify nodes to remove

        while subgraph.number_of_edges() > 50:
            # Find the node with the highest degree
            most_connected_node = max(subgraph.degree, key=lambda x: x[1])[0]

            # Remove this node from the original graph (to update the real graph)
            G.remove_edges_from(list(G.edges(most_connected_node)))

            # Update subgraph to reflect the removal
            subgraph = G.subgraph(c).copy()
    return G
