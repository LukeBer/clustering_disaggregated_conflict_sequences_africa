import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.classes.SEQ_CLUSTER_class import SEQ_CLUSTER


class HDBSCAN_SEQ_CLUSTER(SEQ_CLUSTER):
    def __init__(
        self,
        df: pd.DataFrame,
        MIN_NO_FATALITIES: int = 25,
        HDBSCAN_MIN_SEQ_LENGTH: int = 5,  # For HDBSCAN clustering parameters
        MIN_SEQ_EVENTS: int = 10,  # For filtering sequences
    ):
        """
        Initialize a HDBSCAN-based sequence clustering object.

        This class extends SEQ_CLUSTER and uses HDBSCAN to identify dense clusters of conflict events
        in space and time, which are then interpreted as sequences.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing event-level conflict data. Must contain 'latitude', 'longitude',
            'event_date', 'fatalities', and 'event_id_cnty'.
        MIN_NO_FATALITIES : int, optional
            Minimum number of fatalities for a sequence to be kept (default: 25).
        HDBSCAN_MIN_SEQ_LENGTH : int, optional
            Minimum number of events for HDBSCAN clusters and sequence filtering (default: 5).
        MIN_SEQ_EVENTS : int, optional
            Minimum number of events in a sequence for it to be retained (default: 10).

        Attributes
        ----------
        df : pd.DataFrame
            Original DataFrame with an added 'cluster_label' column indicating HDBSCAN cluster assignments.
        MIN_NO_FATALITIES : int
            Minimum fatalities threshold.
        MIN_EVENT_NO : int
            Minimum number of events threshold.
        HDBSCAN_MIN_SEQ_LENGTH : int
            HDBSCAN minimum cluster size parameter.
        event_seq_mapping : pd.Series
            Series mapping event IDs to sequence labels (cluster names).
        """
        super().__init__(
            "A Spatial-Temporal DBSCAN for data entry clustering algorithm to build sequences."
        )
        self.MIN_NO_FATALITIES = MIN_NO_FATALITIES
        self.MIN_EVENT_NO = (
            MIN_SEQ_EVENTS  # Used for HDBSCAN min_cluster_size and min_samples
        )
        self.HDBSCAN_MIN_SEQ_LENGTH = HDBSCAN_MIN_SEQ_LENGTH  # Used for sequence filtering #TODO: change this logic

        # Prepare geo-temporal features
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        df["days_since_start"] = (df["event_date"] - df["event_date"].min()).dt.days

        geo_time = df[["latitude", "longitude", "days_since_start"]].values.astype(
            "float32"
        )
        geo_time_scaled = StandardScaler().fit_transform(geo_time)

        # Run HDBSCAN
        print("Performing HDBSCAN clustering")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.HDBSCAN_MIN_SEQ_LENGTH,
            min_samples=self.HDBSCAN_MIN_SEQ_LENGTH,
            metric="euclidean",
            approx_min_span_tree=True,
        )

        labels = clusterer.fit_predict(geo_time_scaled)

        df["cluster_label"] = labels

        self._create_event_seq_mapping(df)

        self.df = df

    def _create_event_seq_mapping(self, df: pd.DataFrame):
        """
        Create a mapping from event IDs to HDBSCAN-derived sequence labels.

        This method:
        - Separates noise points (cluster_label == -1) and maps them to 'no_cluster'.
        - Aggregates clustered events to generate descriptive cluster names including date ranges.
        - Filters clusters that do not meet minimum event count or fatalities thresholds.
        - Returns a final Series mapping each event ID to its assigned sequence.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing event-level conflict data with 'cluster_label' assigned by HDBSCAN.

        Notes
        -----
        - Events labeled as 'no_cluster' include both noise points and clusters failing the filtering criteria.
        - Explodes lists of event IDs to create a flat mapping suitable for sequence extraction.
        - Prints warnings if any events are not included in the final mapping.
        """
        # Separate noise points (cluster_label == -1)
        noise_events = df[df["cluster_label"] == -1]["event_id_cnty"]

        # Create no_cluster mapping for noise events
        no_cluster_because_noise = pd.Series(
            noise_events.values, index=noise_events.values, name="cluster_name"
        ).map(lambda _: "no_cluster")

        clustered_events = df[df["cluster_label"] != -1]

        # Create sequence mapper similar to DYAD_YEAR class
        sequence_mapper = (
            clustered_events.groupby("cluster_label")
            .agg(
                {
                    "event_id_cnty": list,
                    "fatalities": "sum",
                    "event_date": lambda x: [
                        min(x),
                        max(x),
                    ],  # Get date range for naming
                }
            )
            .reset_index()
        )

        # Add descriptive names for clusters (similar to dyad_year naming)
        sequence_mapper["date_range"] = sequence_mapper["event_date"].apply(
            lambda dates: f"{dates[0].strftime('%Y-%m') if pd.notna(dates[0]) else 'unknown'}-{dates[1].strftime('%Y-%m') if pd.notna(dates[1]) else 'unknown'}"
        )
        sequence_mapper["cluster_name"] = (
            "cluster_"
            + sequence_mapper["cluster_label"].astype(str)
            + " | "
            + sequence_mapper["date_range"]
        )
        sequence_mapper.drop(
            columns=["event_date", "date_range", "cluster_label"], inplace=True
        )

        # Filter sequences with less than minimum events and min number fatalities
        sequence_mapper.loc[
            sequence_mapper["event_id_cnty"].apply(len) < self.MIN_EVENT_NO,
            "cluster_name",
        ] = "no_cluster"
        sequence_mapper.loc[
            sequence_mapper["fatalities"] < self.MIN_NO_FATALITIES, "cluster_name"
        ] = "no_cluster"
        if (
            len(sequence_mapper.loc[sequence_mapper["cluster_name"] == "no_cluster", :])
            > 0
        ):
            print(
                "WARNING: Some events could not be clustered. Check under 'no_cluster'."
            )

        # Explode and create final mapping
        sequence_mapper = sequence_mapper.explode("event_id_cnty")
        sequence_mapper["event_id_cnty"] = sequence_mapper["event_id_cnty"].astype(str)
        sequence_mapper = sequence_mapper.set_index("event_id_cnty")

        final_mapping = pd.concat([sequence_mapper, no_cluster_because_noise])

        # Check for missing events
        missing = df[~df["event_id_cnty"].isin(final_mapping.index)]
        if not missing.empty:
            print("Some event values are missing in the final mapping index:")
            print(missing["event_id_cnty"].unique())

        self.event_seq_mapping = final_mapping["cluster_name"]
