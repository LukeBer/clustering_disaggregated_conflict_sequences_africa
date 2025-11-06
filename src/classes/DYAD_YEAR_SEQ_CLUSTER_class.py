import pandas as pd

from src.classes.SEQ_CLUSTER_class import SEQ_CLUSTER


class DYAD_YEAR_SEQ_CLUSTER(SEQ_CLUSTER):
    def __init__(
        self,
        df: pd.DataFrame,
        continuous: bool = False,
        MIN_SEQ_EVENTS: int = 10,  # For filtering sequences
        MIN_NUMBER_FATALITIES: int = 25,
    ):
        """
        Initialize a dyad-year-based sequence clustering object.

        One sequence is defined for each dyad over time.
        A dyad is only considered active if it meets a minimum number of fatalities.
        Sequences can optionally be split if a dyad is not continuously active.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with conflict events. Must contain columns:
            'dyad', 'year', 'event_id_cnty', and 'fatalities'.
        continuous : bool, optional
            If True, break sequences when dyad activity is non-continuous across years.
        MIN_SEQ_EVENTS : int, optional
            Minimum number of events required for a sequence to be retained (default: 10).
        MIN_NUMBER_FATALITIES : int, optional
            Minimum number of fatalities per dyad-year to consider it active (default: 25).

        Attributes
        ----------
        event_seq_mapping : pd.Series
            Mapping of event IDs to their dyad-year sequence labels.
        continuous : bool
            Indicates whether sequences are split on inactivity gaps.
        MIN_EVENT_NO : int
            Minimum number of events per sequence.
        MIN_NUMBER_FATALITIES : int
            Minimum fatalities per dyad-year to be included in a sequence.
        """
        super().__init__(
            "One unit is based on one dyad combination and traced over time. A dyad is active as long as it causes 25 fatalities. "
            "There is an option to break a sequence if a dyad is not continuously active."
        )
        self.continuous = continuous
        self.MIN_EVENT_NO = MIN_SEQ_EVENTS  # Map to base class property
        self.MIN_NUMBER_FATALITIES = MIN_NUMBER_FATALITIES

        grouped_data = (
            df.groupby(["dyad", "year"])
            .agg({"event_id_cnty": list, "fatalities": "sum"})
            .rename(columns={"fatalities": "fatalities_count"})
            .reset_index()
        )

        # Filter for groups with more than 25 fatalities
        inactive_data = grouped_data[
            grouped_data["fatalities_count"] < MIN_NUMBER_FATALITIES
        ]
        grouped_data = grouped_data[
            grouped_data["fatalities_count"] >= MIN_NUMBER_FATALITIES
        ]
        no_cluster_because_inactive = pd.Series(
            inactive_data.explode("event_id_cnty")["event_id_cnty"].values,
            index=inactive_data.explode("event_id_cnty")["event_id_cnty"].values,
            name="dyad_year",
        ).map(lambda _: "no_cluster")

        # Handle sequenced dyads
        if self.continuous:
            clustered_data = grouped_data.groupby("dyad", group_keys=False).apply(
                self.assign_block_ids
            )

            sequence_mapper = (
                clustered_data.groupby(["dyad", "block_id"])
                .agg(
                    {
                        "year": list,
                        "event_id_cnty": lambda x: sum(x, []),
                    }
                )
                .reset_index()
                .drop(columns="block_id")
            )
        else:
            sequence_mapper = (
                grouped_data.groupby(["dyad"])
                .agg(
                    {
                        "year": list,
                        "event_id_cnty": lambda x: sum(x, []),
                    }
                )
                .reset_index()
            )

        # Add year range label to sequenced ones
        sequence_mapper["year_range"] = sequence_mapper["year"].apply(
            lambda yrs: f"{min(yrs)}–{max(yrs)}" if len(yrs) > 1 else str(yrs[0])
        )
        sequence_mapper["dyad_year"] = (
            sequence_mapper["dyad"] + " | " + sequence_mapper["year_range"]
        )
        sequence_mapper.drop(columns=["year", "dyad", "year_range"], inplace=True)

        # sequences with less then 5 events are considered no cluster
        sequence_mapper.loc[
            sequence_mapper["event_id_cnty"].apply(len) < self.MIN_EVENT_NO, "dyad_year"
        ] = "no_cluster"
        if (
            len(sequence_mapper.loc[sequence_mapper["dyad_year"] == "no_cluster", :])
            > 0
        ):
            print(
                "WARNING: Some events could not be clustered. Check under 'no_cluster'."
            )

        # Explode and concat both mappings
        sequence_mapper = sequence_mapper.explode("event_id_cnty")
        sequence_mapper["event_id_cnty"] = sequence_mapper["event_id_cnty"].astype(str)
        sequence_mapper = sequence_mapper.set_index("event_id_cnty")

        final_mapping = pd.concat([sequence_mapper, no_cluster_because_inactive])

        missing = df[~df["event_id_cnty"].isin(final_mapping.index)]
        if not missing.empty:
            print("Some event values are missing in the final mapping index:")
            print(missing["event_id_cnty"].unique())

        self.event_seq_mapping = final_mapping["dyad_year"]

    def assign_block_ids(self, group):
        """
        Assign block IDs to a dyad's events to split sequences at gaps in consecutive years.

        Parameters
        ----------
        group : pd.DataFrame
            Subset of events for a single dyad, must contain a 'year' column.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with an additional 'block_id' column indicating continuous sequences.
        """
        group = group.sort_values("year")
        year_diff = group["year"].diff().fillna(1)
        block_id = (year_diff != 1).cumsum()
        group["block_id"] = block_id
        return group
