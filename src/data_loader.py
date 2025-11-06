import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import AFRICAN_COUNTRIES, END_DATE, START_DATE
from src.utils.geographic import lat_lon_to_prio_gid

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data/raw_data"
DATA_DIR_PROCESSED = PROJECT_DIR / "data/processed_data"

actor_mapping = {
    "Al Shabaab": "Al-Shabaab",
    "Anti-Balaka": "anti-Balaka",
    "Dan Na Ambassagou": "Dan na Ambassagou",
    "BRSC: Shura Council of Benghazi Revolutionaries": "Benghazi Revolutionaries Shura Council",
    "HI: Hizbul Islam": "Hizbul Islam",
    "Ansar Beit al-Maqdis": "Ansar Bayt al-Maqdis",
    "Al-Gamaa al-Islamiyya": "al-Gama'a al-Islamiyya",
    "Ganda Izo": "Ganda Iso",
    "Jamaa Ansar al-Islam": "Jama'at Ansar al-Islam",
    "Government of Sudan (2019-)": "Government of Sudan",
    "Government of South Sudan (2011-)": "Government of South Sudan",
    "Government of Ethiopia (2018-)": "Government of Ethiopia",
    "Government of Kenya (2022-)": "Government of Kenya",
    "Government of Burundi (2005-)": "Government of Burundi",
    "Government of the Central African Republic (20...)": "Government of Central African Republic",
    "Government of Cameroon (1982-)": "Government of Cameroon",
    "Government of Egypt (2014-)": "Government of Egypt",
    "Government of Mozambique (1990-)": "Government of Mozambique",
    "Government of Angola (1975-)": "Government of Angola",
    "Government of Benin (2016-)": "Government of Benin",
    "Government of Uganda (1986-)": "Government of Uganda",
    "Government of Chad (2021-)": "Government of Chad",
    "Government of Burkina Faso (2015-2022)": "Government of Burkina Faso",
    "Government of Morocco (1999-)": "Government of Morocco",
    "Government of Algeria (2019-)": "Government of Algeria",
    "Government of Eritrea (1993-)": "Government of Eritrea",
    "Government of Burkina Faso (1987-2014)": "Government of Burkina Faso",
    "Government of Mali (2021-)": "Government of Mali",
    "Government of Djibouti (1999-)": "Government of Djibouti",
    "Government of Sierra Leone (1998-2007)": "Government of Sierra Leone",
    "Government of Rwanda (1994-)": "Government of Rwanda",
    "Government of Burkina Faso (2022-)": "Government of Burkina Faso",
    "Government of Nigeria (2023-)": "Government of Nigeria",
    "Government of South Africa (1994-2024)": "Government of South Africa",
    "Government of Guinea-Bissau (2020-)": "Government of Guinea-Bissau",
    "Government of the Ivory Coast (2011-)": "Government of Ivory Coast",
    "Government of Tanzania (1964-)": "Government of Tanzania",
    "Government of Somalia (2022-)": "Government of Somalia",
    "Government of Guinea (2021-)": "Government of Guinea",
    "Government of Togo (2005-)": "Government of Togo",
    "Government of Sierra Leone (2018-)": "Government of Sierra Leone",
    "Government of Tunisia (2019-)": "Government of Tunisia",
    "Government of Gambia (2017-)": "Government of Gambia",
    "Government of Burkina Faso (2014-2015)": "Government of Burkina Faso",
    "Government of Mauritania (2019-)": "Government of Mauritania",
    "Government of Madagascar (2019-)": "Government of Madagascar",
    "Government of Mauritania (2009-2019)": "Government of Mauritania",
    "Government of Sierra Leone (2007-2018)": "Government of Sierra Leone",
    "Government of the Central African Republic (19...)": "Government of Central African Republic",
    "Government of Lesotho (2017-)": "Government of Lesotho",
    "Government of Guinea-Bissau (2014-2020)": "Government of Guinea-Bissau",
    "Government of Madagascar (2014-2019)": "Government of Madagascar",
    "Government of Liberia (2018-)": "Government of Liberia",
}


def data_preprocessing(
    df,
    min_time_prec=2,
    country_filter=AFRICAN_COUNTRIES,
    date_start=START_DATE,
    date_end=END_DATE,
):
    """
    Preprocess conflict event data from ACLED/UCDP sources.

    Steps include:
    - Filtering by country and date range
    - Filtering by minimum time precision
    - Adding dyad identifiers
    - Mapping coordinates to PRIO-GRID GIDs
    - Enriching with geographic/environmental covariates

    Parameters
    ----------
    df : pd.DataFrame
        Raw event-level data.
    dyad_id : bool, default=True
        Whether to create dyad identifiers using actor IDs.
        If False, dyads are created based on actor names.
    min_time_prec : int, optional
        Maximum allowed time precision (1=day, 2=week, etc.).
    country_filter : list[str], optional
        List of countries to include.
    date_start : str or pd.Timestamp, optional
        Start date for filtering events.
    date_end : str or pd.Timestamp, optional
        End date for filtering events.

    Returns
    -------
    pd.DataFrame
        Preprocessed and enriched data including dyad identifiers,
        PRIO-GRID IDs, and environmental/geographic features.
    """
    if country_filter is not None:
        df_filtered = df[df["country"].isin(country_filter)].copy()

    if date_start and date_end is not None:
        df_filtered = df_filtered[
            (df_filtered["event_date"] >= date_start)
            & (df_filtered["event_date"] <= date_end)
        ]

    if min_time_prec is not None:
        df_filtered = df_filtered[df_filtered["time_precision"] <= min_time_prec]
        if min_time_prec <= 2:
            df_filtered.loc[:, "month"] = pd.DatetimeIndex(
                df_filtered["event_date"]
            ).month
            df_filtered.loc[:, "month_year"] = pd.to_datetime(
                df_filtered[["year", "month"]].assign(day=1)
            )

    df_filtered["dyad"] = create_dyad_with_id(df_filtered)

    df_filtered.loc[:, "gid"] = lat_lon_to_prio_gid(
        df_filtered["latitude"].values, df_filtered["longitude"].values
    )
    df_enriched = enrich_data(df_filtered)
    return df_enriched


def get_combined_data():
    """
    Load and combine ACLED and UCDP event datasets for Africa.

    - Standardizes column names to ACLED schema
    - Maps actor names to standardized form
    - Maps disorder types and time precision
    - Returns a single combined DataFrame

    Returns
    -------
    pd.DataFrame
        Combined ACLED/UCDP data with standardized columns and actor mappings.

    Raises
    ------
    ValueError
        If ACLED or UCDP source files are not found.
    """
    acled_df = None
    ucdp_df = None

    # Load ACLED data
    acled_path = os.path.join(
        DATA_DIR,
        "1997-01-01-2023-12-31-Eastern_Africa-Middle_Africa-Northern_Africa-Southern_Africa-Western_Africa.csv",
    )
    try:
        acled_df = pd.read_csv(acled_path)
    except FileNotFoundError:
        raise ValueError("ACLED data not accessible, path not found.")

    # Load UCDP data
    ucdp_path = os.path.join(DATA_DIR, "GEDEvent_v25_1.csv")
    try:
        ucdp_df = pd.read_csv(ucdp_path)
    except FileNotFoundError:
        raise ValueError("UCDP data not accessible, path not found.")

    # Column mapping UCDP → ACLED schema
    ucdp_to_acled = {
        "id": "event_id_cnty",
        "date_start": "event_date",
        "year": "year",
        "date_prec": "time_precision",
        "type_of_violence": "disorder_type",  # numeric codes → unify later
        "side_a": "actor1",
        "side_b": "actor2",
        "side_a_dset_id": "actor1_id",
        "side_b_dset_id": "actor2_id",
        "latitude": "latitude",
        "longitude": "longitude",
        "adm_1": "admin1",
        "adm_2": "admin2",
        "where_description": "location",
        "country": "country",
        "region": "region",
        "where_prec": "geo_precision",
        "best": "fatalities",
        "source_article": "notes",
        "source_original": "source",
    }

    columns = list(ucdp_to_acled.values())

    acled_df.loc[:, "actor1_id"] = acled_df.loc[:, "actor1"]
    acled_df.loc[:, "actor2_id"] = acled_df.loc[:, "actor2"]

    # Standardize column names
    acled_std = acled_df[columns].copy()
    ucdp_std = ucdp_df.rename(columns=ucdp_to_acled)[list(ucdp_to_acled.values())]

    acled_std.loc[:, "source_dataset"] = "ACLED"
    ucdp_std.loc[:, "source_dataset"] = "UCDP"
    ucdp_std.loc[:, "event_date"] = pd.to_datetime(
        ucdp_std["event_date"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
    )
    acled_std.loc[:, "event_date"] = pd.to_datetime(
        acled_std["event_date"], errors="coerce"
    )

    combined = pd.concat([acled_std, ucdp_std], ignore_index=True, sort=False)

    combined["event_id_cnty"] = combined["event_id_cnty"].astype(str)
    combined.loc[:, "actor1"] = (
        combined["actor1"].map(actor_mapping).fillna(combined["actor1"])
    )
    combined.loc[:, "actor2"] = (
        combined["actor2"].map(actor_mapping).fillna(combined["actor2"])
    )

    violence_map = {
        1: "Political violence",
        2: "Demonstrations",
        3: "Strategic developments",
    }
    combined.loc[:, "disorder_type"] = (
        combined["disorder_type"].map(violence_map).fillna(combined["disorder_type"])
    )

    precision_map = {1: "day", 2: "week", 3: "month", 4: "month_range", 5: "year_range"}
    combined.loc[:, "time_precision_text"] = combined["time_precision"].map(
        precision_map
    )

    country_name_mapping = {
        "DR Congo (Zaire)": "Democratic Republic of Congo",
        "eSwatini": "Eswatini",
        "Kingdom of eSwatini (Swaziland)": "Eswatini",
        "Madagascar (Malagasy)": "Madagascar",
        "Republic of Congo": "Congo",
    }
    combined.loc[:, "country"] = combined["country"].replace(country_name_mapping)

    return combined


def create_dyad_with_id(df):
    """
    Generate dyad identifiers using numeric actor IDs.

    Parameters
    ----------
    df : pd.DataFrame
        Event-level data with 'actor1' and 'actor2' columns.

    Returns
    -------
    pd.Series
        Dyad identifiers of the form "actor1_id:actor2_id".
    """
    actor_names = (
        pd.concat(
            [
                df[["actor1"]].dropna().rename(columns={"actor1": "actor_name"}),
                df[["actor2"]].dropna().rename(columns={"actor2": "actor_name"}),
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    actor_names["actor_id"] = ["{:05d}".format(i + 1) for i in range(len(actor_names))]

    name_to_id = dict(zip(actor_names["actor_name"], actor_names["actor_id"]))
    df["actor1_id"] = df["actor1"].map(name_to_id)
    df["actor2_id"] = df["actor2"].map(name_to_id)
    df["actor2_id"] = df["actor2_id"].fillna(df["actor1_id"])

    dyad_min = df[["actor1_id", "actor2_id"]].min(axis=1).astype(str)
    dyad_max = df[["actor1_id", "actor2_id"]].max(axis=1).astype(str)
    df["dyad"] = dyad_min + ":" + dyad_max
    return df["dyad"]


def get_actor_dyad_mapping(df_africa, exclude_special_actors=True):
    """
    Construct a mapping of actors to dyads in which they participate.

    Parameters
    ----------
    df_africa : pd.DataFrame
        African event data containing dyads and actor IDs.
    exclude_special_actors : bool, default=True
        Whether to exclude civilians, government actors, and extreme outliers.

    Returns
    -------
    pd.DataFrame
        Mapping of actor IDs to the list of dyads they are involved in,
        filtered to exclude single-dyad actors and optionally special actors.
    """
    if exclude_special_actors:
        civilian_id_list = df_africa[
            df_africa["actor2"].str.contains("Civilians", na=False)
        ]["actor2_id"].unique()

        government_id_list = np.unique(
            np.concatenate(
                [
                    df_africa[
                        df_africa["actor1"].str.contains(
                            "Government|Police|Military Forces", na=False
                        )
                    ]["actor1_id"].values,
                    df_africa[
                        df_africa["actor2"].str.contains(
                            "Government|Police|Military Forces", na=False
                        )
                    ]["actor2_id"].values,
                ]
            )
        )

        outlier = df_africa[
            df_africa["actor2"].str.contains(
                "Al-Shabaab|Unidentified|Boko Haram|Group for Support of Islam and Muslims",
                na=False,
            )
        ]["actor2_id"].unique()

    actor_df = pd.melt(
        df_africa[["dyad", "actor1_id", "actor2_id"]],
        id_vars=["dyad"],
        value_vars=["actor1_id", "actor2_id"],
        value_name="actor",
    )
    actor_to_dyads_df = actor_df.groupby("actor")["dyad"].agg(list).reset_index()
    actor_to_dyads_df = actor_to_dyads_df[actor_to_dyads_df["dyad"].apply(len) > 1]
    actor_to_dyads_df = actor_to_dyads_df[
        ~actor_to_dyads_df["actor"].isin(civilian_id_list)
    ]
    actor_to_dyads_df = actor_to_dyads_df[
        ~actor_to_dyads_df["actor"].isin(government_id_list)
    ]
    actor_to_dyads_df = actor_to_dyads_df[~actor_to_dyads_df["actor"].isin(outlier)]
    return actor_to_dyads_df


def enrich_data(df):
    """
    Merge event-level data with PRIO-GRID geographic/environmental data.

    Parameters
    ----------
    df : pd.DataFrame
        Event-level data containing a 'gid' column.

    Returns
    -------
    pd.DataFrame
        Data enriched with PRIO-GRID features, including:
        - Land cover fractions (agri_gc, forest_gc, urban_gc, etc.)
        - Terrain and climate features (mountains_mean, rainseas, ttime_mean)
        - Distances to borders and capitals
    """
    prio_grid_path = os.path.join(DATA_DIR, "PRIO-GRID_Static_bdist1_capdist(2004).csv")
    try:
        prio_grid = pd.read_csv(prio_grid_path)
    except FileNotFoundError:
        raise ValueError("PRIO-GRID data not accessible, path not found.")

    df_merged = df.merge(prio_grid, on="gid", how="left")

    usable_cols = [
        "gid",
        "agri_gc",
        "aquaveg_gc",
        "barren_gc",
        "forest_gc",
        "herb_gc",
        "shrub_gc",
        "urban_gc",
        "water_gc",
        "mountains_mean",
        "rainseas",
        "ttime_mean",
        "dist_border_km_2004",
        "dist_capital_km_2004",
    ]
    old_columns = list(df.columns)  # convert Index -> list
    combined_cols = list(dict.fromkeys(old_columns + usable_cols))

    df_usable = df_merged[combined_cols].copy()
    return df_usable
