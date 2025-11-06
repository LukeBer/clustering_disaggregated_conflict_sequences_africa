import copy
from typing import Dict

import numpy as np


def normalize_seq_cluster(seq_cluster_instance, normalization_config: Dict):
    """
    Normalize sequences in a SEQ_CLUSTER instance according to a specified configuration.

    Parameters
    ----------
    seq_cluster_instance : SEQ_CLUSTER
        Instance containing sequences to normalize.
    normalization_config : dict
        Dictionary specifying normalization settings for each metric and temporal curve options.
        Example:
        {
            'fatalities': 'proportional',          # 'proportional', 'minmax', 'none'
            'event_id_cnty': 'proportional',       # 'proportional', 'minmax', 'none'
            'time': 'temporal_metrics',            # 'temporal_metrics', 'none'
            'temporal_curve': 'fatality_curve',    # 'fatality_curve', 'none'
            'n_bins': 20,                          # number of temporal bins
            'binning_method': 'temporal',          # 'temporal', 'event_count'
            'curve_type': 'cumulative',            # 'cumulative', 'relative'
            'temporal_curve_integral': 'integral'  # 'integral', 'none'
        }

    Raises
    ------
    ValueError
        If required metrics are missing from the sequence dictionary.
    """
    _check_sequence_and_metric_available(seq_cluster_instance)

    normalized_instance = copy.copy(seq_cluster_instance)

    normalize_sequences(
        normalized_instance.sequence_dict,
        normalization_config,
        seq_cluster_instance.index_name,
    )


def _check_sequence_and_metric_available(instance):
    """
    Ensure that a SEQ_CLUSTER instance contains required metrics for normalization.

    Parameters
    ----------
    instance : SEQ_CLUSTER
        Cluster instance to check.

    Raises
    ------
    ValueError
        If sequence_dict is empty or required metrics ('fatalities', 'gid', 'event_id_cnty') are missing.
    """

    if not instance.sequence_dict:
        raise ValueError(
            "No sequences found in the cluster instance. Run extract_sequences first."
        )

    for metric in ["fatalities", "gid", "event_id_cnty"]:
        if metric not in instance.sequence_dict:
            raise ValueError(f"Metric '{metric}' not found in sequence.")


def normalize_sequences(seq_dict: Dict, config: Dict, index_name: str) -> Dict:
    """
    Apply normalization to metrics and temporal curves in a sequence dictionary.

    Parameters
    ----------
    seq_dict : dict
        Dictionary containing sequences for metrics: 'fatalities', 'event_id_cnty', 'gid', 'index'.
    config : dict
        Normalization configuration dictionary (see `normalize_seq_cluster`).
    index_name : str
        Name of the index column (e.g., 'event_date', 'month_year') for temporal normalization.

    Returns
    -------
    dict
        Sequence dictionary with additional normalized fields.
    """

    cumulative = config.get("curve_type", "cumulative") == "cumulative"

    normalized_index = [normalize_time(seq, index_name) for seq in seq_dict["index"]]

    if config.get("fatalities") != "none":
        normalized_fatalities = [
            normalize_metrics(seq, config["fatalities"], cumulative)
            for seq in seq_dict["fatalities"]
        ]
        seq_dict["normalized_fatalities"] = normalized_fatalities

    if config.get("event_id_cnty") != "none":
        normalized_event_id_cnty = [
            normalize_metrics(seq, config["event_id_cnty"], cumulative)
            for seq in seq_dict["event_id_cnty"]
        ]
        seq_dict["normalized_event_id_cnty"] = normalized_event_id_cnty

    if config.get("time", "none") != "none":
        seq_dict["normalized_index"] = normalized_index

    if config.get("temporal_curve", "none") != "none":

        n_bins = config.get("n_bins", 20)
        binning_method = config.get("binning_method", "temporal")

        results = [
            normalize_temporal_curve(
                index, fat, gid, e_count, n_bins, binning_method, cumulative
            )
            for index, fat, gid, e_count in zip(
                normalized_index,
                seq_dict["fatalities"],
                seq_dict["gid"],
                seq_dict["event_id_cnty"],
            )
        ]
        binned_index, binned_fatalities, binned_grid, binned_event_count = zip(*results)
        seq_dict["binned_fatalities"] = binned_fatalities
        seq_dict["binned_index"] = binned_index
        seq_dict["binned_grid"] = binned_grid
        seq_dict["binned_event_id_cnty"] = binned_event_count

    if config.get("temporal_curve_integral", "none") != "none":

        binned_fatalities_integral = [
            integral_temporal_curve(index, fat, n_bins)
            for index, fat in zip(normalized_index, seq_dict["fatalities"])
        ]
        seq_dict["binned_fatalities_integral"] = binned_fatalities_integral


def normalize_metrics(
    values: np.ndarray, method: str = "proportional", cumulative: bool = False
) -> np.ndarray:
    """
    Normalize a numeric array according to a chosen method.

    Parameters
    ----------
    values : np.ndarray
        Numeric array to normalize.
    method : str, default='proportional'
        Normalization method: 'proportional' or 'minmax'.
    cumulative : bool, default=False
        If True, returns the cumulative sum of normalized values.

    Returns
    -------
    np.ndarray
        Normalized array.

    Raises
    ------
    ValueError
        If an unknown normalization method is specified.
    """
    if method == "proportional":
        total = values.sum()
        norm = values / total if total > 0 else values

    elif method == "minmax":
        min_v, max_v = values.min(), values.max()
        norm = (
            (values - min_v) / (max_v - min_v)
            if max_v > min_v
            else np.zeros_like(values)
        )

    else:
        raise ValueError(f"Unknown normalization method: '{method}'")

    if cumulative:
        norm = np.cumsum(norm)

    return norm


def convert_index_to_relative(indices, index_name):
    """
    Convert datetime or numeric indices to relative numeric indices starting at zero.

    Parameters
    ----------
    indices : array-like
        Sequence of datetime or numeric indices.
    index_name : str
        Type of index: 'month_year', 'event_date', or numeric.

    Returns
    -------
    np.ndarray
        Array of relative indices.

    Raises
    ------
    ValueError
        If indices are not numeric or datetime-like.
    """
    start = min(indices)

    if index_name == "month_year":
        return np.array(
            [(d.year - start.year) * 12 + (d.month - start.month) for d in indices]
        )

    elif index_name == "event_date":
        return np.array([(d - start).days for d in indices])

    else:
        try:
            return np.array([i - start for i in indices])
        except TypeError:
            raise ValueError(
                "Index is not numeric or datetime-like. Cannot convert to relative values."
            )


def normalize_time(indices: np.ndarray, index_name: str) -> np.ndarray:
    """
    Normalize time indices to range [0, 1].

    Parameters
    ----------
    indices : np.ndarray
        Sequence of datetime or numeric indices.
    index_name : str
        Type of index, e.g., 'event_date' or 'month_year'.

    Returns
    -------
    np.ndarray
        Normalized time array in [0, 1].
    """
    relative_index = convert_index_to_relative(indices, index_name)
    min_i, max_i = relative_index.min(), relative_index.max()
    norm = (
        (relative_index - min_i) / (max_i - min_i)
        if max_i > min_i
        else np.zeros_like(relative_index)
    )

    return norm


def normalize_spatial(grid_counts: np.ndarray, method: str) -> Dict:
    """
    Compute spatial normalization metrics from grid counts.

    Parameters
    ----------
    grid_counts : np.ndarray
        Count of events per spatial grid cell.
    method : str
        Normalization method. Currently supports 'spread'.

    Returns
    -------
    dict
        Dictionary containing:
        - 'grid_count_max_prop': maximum proportion in a cell
        - 'grid_count_variance_prop': variance of normalized counts
    """
    result = {}

    if method == "spread":
        total = grid_counts.sum()
        norm = grid_counts / total if total > 0 else grid_counts
        result.update(
            {
                "grid_count_max_prop": norm.max() if total > 0 else 0,
                "grid_count_variance_prop": np.var(norm) if total > 0 else 0,
            }
        )

    return norm


def normalize_temporal_curve(
    relative_index: np.ndarray,
    fatalities: np.ndarray,
    grid: np.ndarray,
    event_counts: np.ndarray,
    n_bins: int = 20,
    binning_method: str = "temporal",
    cumulative: bool = False,
) -> Dict:
    """
    Aggregate sequences into binned temporal curves.

    Parameters
    ----------
    relative_index : np.ndarray
        Normalized relative time indices.
    fatalities : np.ndarray
        Fatality counts corresponding to the sequence.
    grid : np.ndarray
        Grid counts for spatial aggregation.
    event_counts : np.ndarray
        Event counts per sequence.
    n_bins : int, default=20
        Number of temporal bins.
    binning_method : str, default='temporal'
        'temporal' for regular time bins, 'event_count' for equal-event bins.
    cumulative : bool, default=False
        Whether to produce cumulative curves.

    Returns
    -------
    tuple of np.ndarray
        (binned_index, binned_fatalities, binned_grid, binned_event_count)
    """

    if binning_method == "temporal":
        # Regular bins over time
        bins = np.linspace(relative_index.min(), relative_index.max(), n_bins + 1)
        bin_indices = np.digitize(relative_index, bins) - 1
    else:
        # Equal-event binning: sort by time, then split into bins with ≈ same number of events
        bin_size = max(1, event_counts.apply(np.sum) // n_bins)
        # Expand time and fatalities by event count (one row per event)
        expanded_times = np.repeat(relative_index, event_counts)

        # Sort by time and assign bin index
        sort_idx = np.argsort(expanded_times)
        bin_indices = np.repeat(np.arange(n_bins), bin_size)[: len(expanded_times)]

    # Aggregate fatalities per bin

    binned_fatalities = np.zeros(n_bins)
    binned_grid = np.zeros(n_bins)
    binned_event_count = np.zeros(n_bins)
    for i in range(n_bins):
        binned_fatalities[i] = fatalities[bin_indices == i].sum()
        binned_grid[i] = grid[bin_indices == i].sum()
        binned_event_count[i] = event_counts[bin_indices == i].sum()

    binned_fatalities = normalize_metrics(binned_fatalities)
    binned_event_count = normalize_metrics(binned_event_count)

    if cumulative:
        binned_fatalities = np.cumsum(binned_fatalities)
        binned_event_count = np.cumsum(binned_event_count)

    binned_index = np.arange(n_bins)

    return binned_index, binned_fatalities, binned_grid, binned_event_count


def integral_temporal_curve(
    time: np.ndarray, fatalities: np.ndarray, n_bins: int = 20
) -> Dict:
    """
    Compute the integral of normalized fatalities over temporal bins.

    Parameters
    ----------
    time : np.ndarray
        Normalized time indices.
    fatalities : np.ndarray
        Fatalities sequence to integrate.
    n_bins : int, default=20
        Number of bins for integration.

    Returns
    -------
    np.ndarray
        Array of integrated fatalities per temporal bin.
    """
    norm_fatalities = normalize_metrics(fatalities)
    cumul_norm_fatalities = np.cumsum(norm_fatalities)

    # only temporal makes sense here
    bins = np.linspace(time.min(), time.max(), n_bins + 1)
    binned_fatalities_integrals = np.zeros(n_bins)

    # Step 4: Integrate within each bin using interpolation
    for i in range(n_bins):
        t0, t1 = bins[i], bins[i + 1]

        # Mask points in bin, including points on the edges
        mask = (time >= t0) & (time <= t1)
        t_in_bin = time[mask]
        f_in_bin = cumul_norm_fatalities[mask]

        # Ensure start/end points are included (linear interpolation if needed)
        if t_in_bin.size == 0 or t_in_bin[0] > t0:
            t_in_bin = np.insert(t_in_bin, 0, t0)
            f_start = np.interp(t0, time, cumul_norm_fatalities)
            f_in_bin = np.insert(f_in_bin, 0, f_start)
        if t_in_bin[-1] < t1:
            t_in_bin = np.append(t_in_bin, t1)
            f_end = np.interp(t1, time, cumul_norm_fatalities)
            f_in_bin = np.append(f_in_bin, f_end)

        # Integrate the interpolated segment
        binned_fatalities_integrals[i] = np.trapezoid(f_in_bin, t_in_bin)

    return binned_fatalities_integrals
