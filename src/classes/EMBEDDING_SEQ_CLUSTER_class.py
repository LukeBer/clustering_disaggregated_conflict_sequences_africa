import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from src.classes.SEQ_CLUSTER_class import SEQ_CLUSTER
from src.data_loader import DATA_DIR_PROCESSED


class EMBEDDING_SEQ_CLUSTER(SEQ_CLUSTER):
    def __init__(
        self,
        df: pd.DataFrame,
        HDBSCAN_MIN_SEQ_LENGTH: int = 5,  # For embedding clustering parameters
        MIN_NO_FATALITIES: int = 25,
        MIN_SEQ_EVENTS: int = 10,  # For filtering sequences
        force_recompute: bool = False,
        force_recompute_embeddings: bool = False,
        embedding_file: str = "text_embeddings.npy",
        processed_data_file: str = "embedding_data_seq.csv",
    ):
        """
        Initialize a spatial-temporal sequence clustering class that combines HDBSCAN with text embeddings.

        Parameters
        ----------
        df : pd.DataFrame
            Event-level dataset. Must include columns: 'latitude', 'longitude', 'event_date', 'fatalities',
            'event_id_cnty', 'disorder_type', 'actor1', 'actor2', and 'notes'.
        HDBSCAN_MIN_SEQ_LENGTH : int, optional
            Minimum size of HDBSCAN clusters, default 5.
        MIN_NO_FATALITIES : int, optional
            Minimum number of fatalities required for a cluster to be retained, default 25.
        MIN_SEQ_EVENTS : int, optional
            Minimum number of events required for a cluster to be retained, default 10.
        force_recompute : bool, optional
            If True, recompute processed data and clusters even if files exist.
        force_recompute_embeddings : bool, optional
            If True, recompute text embeddings even if embedding file exists.
        embedding_file : str, optional
            File path to save/load embeddings (default "text_embeddings.npy").
        processed_data_file : str, optional
            File path to save/load processed data with cluster labels (default "embedding_data_seq.csv").

        Attributes
        ----------
        text_embeddings : np.ndarray
            Computed embeddings for each event.
        event_seq_mapping : pd.Series
            Mapping of event IDs to sequence/cluster labels.
        df : pd.DataFrame
            DataFrame with cluster labels and optional processed embeddings.
        """
        super().__init__(
            "A Spatial-Temporal DBSCAN with text embeddings for data entry clustering algorithm to build sequences."
        )

        # Set minimum event number for sequences
        self.MIN_EVENT_NO = MIN_SEQ_EVENTS  # For sequence filtering
        self.MIN_NO_FATALITIES = MIN_NO_FATALITIES
        self.HDBSCAN_MIN_SEQ_LENGTH = HDBSCAN_MIN_SEQ_LENGTH  # For embedding clustering

        if force_recompute:
            if force_recompute_embeddings:
                df = self.calc_embeddings(df, embedding_file)
            else:
                # Try to load existing embeddings
                embeddings_path = DATA_DIR_PROCESSED / embedding_file
                if embeddings_path.exists():
                    print(f"Loading existing embeddings from {embeddings_path}")
                    self.text_embeddings = np.load(embeddings_path)
                else:
                    print("No existing embeddings found, computing new.")
                    df = self.calc_embeddings(df, embedding_file)
            # Calculate sequence IDs
            df = self._create_sequences_lables_with_HDBScan(df, embedding_file)
        else:
            embeddings_path = DATA_DIR_PROCESSED / embedding_file
            if embeddings_path.exists():
                print(f"Loading existing embeddings from {embeddings_path}")
                self.text_embeddings = np.load(embeddings_path)

            # Try to load existing processed data
            processed_data_path = DATA_DIR_PROCESSED / processed_data_file
            if processed_data_path.exists():
                print(f"Loading existing processed data from {processed_data_path}")
                df = pd.read_csv(processed_data_path)

                # Ensure event_date is parsed as datetime if it exists
                if "event_date" in df.columns:
                    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

            else:
                print("No existing processed data found, computing from scratch.")
                df = self.calc_embeddings(df, embedding_file)
                df = self._create_sequences_lables_with_HDBScan(df, embedding_file)

        # Create event_seq_mapping from cluster results
        self.event_seq_mapping = self._create_event_seq_mapping(df)
        self.df = df

    def _create_sequences_lables_with_HDBScan(
        self, df: pd.DataFrame, embedding_file: str
    ):
        """
        Perform HDBSCAN clustering on combined geo-temporal and embedding features to assign sequence labels.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with event-level data and optional embeddings.
        embedding_file : str
            Path to save/load embeddings.

        Returns
        -------
        pd.DataFrame
            DataFrame with a new 'cluster_label' column representing sequence clusters.
        """
        # Load or use existing embeddings
        if hasattr(self, "text_embeddings"):
            text_emb = self.text_embeddings
        else:
            embeddings_path = DATA_DIR_PROCESSED / embedding_file

            if embeddings_path.exists():
                text_emb = np.load(embeddings_path)
                self.text_embeddings = text_emb
            else:
                print("No embeddings found, calculating them first...")
                df = self.calc_embeddings(df, embedding_file)
                text_emb = self.text_embeddings

        # Reduce dimensionality (from 768 to 32 as in embedding.ipynb)
        pca = PCA(n_components=128)
        text_emb_reduced = pca.fit_transform(text_emb.astype("float32"))


        explained_var = pca.explained_variance_ratio_
        cum_var = explained_var.cumsum()

        # nice summary table
        summary = pd.DataFrame({
            'PC': range(1, len(explained_var)+1),
            'Explained_Variance': explained_var,
            'Cumulative': cum_var
        })
        print(summary)

        # Prepare geo-temporal features
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        df["days_since_start"] = (df["event_date"] - df["event_date"].min()).dt.days

        geo_time = df[["latitude", "longitude", "days_since_start"]].values.astype(
            "float32"
        )
        geo_time_scaled = StandardScaler().fit_transform(geo_time)

        # Combine features
        X = np.hstack([text_emb_reduced, geo_time_scaled])
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

        um = UMAP(n_neighbors=15, min_dist=0.0,
               n_components=32, metric="cosine")
        X = um.fit_transform(X)

        # Perform clustering
        print("Performing HDBSCAN clustering")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.HDBSCAN_MIN_SEQ_LENGTH,  # Use clustering parameter, not sequence filtering
            min_samples=3,
            metric="euclidean",
            approx_min_span_tree=True,
        )
        labels = clusterer.fit_predict(X)

        # Add cluster labels to dataframe
        df["cluster_label"] = labels

        processed_data_path = DATA_DIR_PROCESSED / "embedding_data_seq.csv"
        df.to_csv(processed_data_path, index=False)
        print(f"Processed data saved to {processed_data_path}")

        return df

    def _create_event_seq_mapping(self, df: pd.DataFrame):
        """
        Generate a mapping from event IDs to HDBSCAN-derived sequence labels.

        Steps:
        - Separates noise points (cluster_label == -1) and assigns 'no_cluster'.
        - Aggregates clustered events to create descriptive cluster names with date ranges.
        - Filters clusters below minimum event count or fatalities.
        - Explodes lists of event IDs into a flat Series mapping.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'cluster_label' column.

        Returns
        -------
        pd.Series
            Mapping of each event_id_cnty to its cluster/sequence label.
        """
        if "cluster_label" not in df.columns:
            raise ValueError("cluster_label column not found. Run calc_seq_ids first.")

        # Separate noise points (cluster_label == -1)
        noise_events = df[df["cluster_label"] == -1]["event_id_cnty"]
        clustered_events = df[df["cluster_label"] != -1]

        # Create sequence mapper similar to DYAD_YEAR class
        sequence_mapper = (
            clustered_events.groupby("cluster_label")
            .agg(
                {
                    "event_id_cnty": list,
                    "event_date": lambda x: [min(x), max(x)],
                    "fatalities": "sum",
                }
            )
            .reset_index()
        )

        sequence_mapper["date_range"] = sequence_mapper["event_date"].apply(
            self._format_date_range
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

        # Create no_cluster mapping for noise events
        no_cluster_because_noise = pd.Series(
            noise_events.values, index=noise_events.values, name="cluster_name"
        ).map(lambda _: "no_cluster")

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

        return final_mapping["cluster_name"]

    def _format_date_range(self, dates):
        """
        Safely format a start and end date into a string representation.

        Parameters
        ----------
        dates : list-like
            List or array containing [start_date, end_date], either as datetime or string.

        Returns
        -------
        str
            Formatted date range string 'YYYY-MM–YYYY-MM' or 'unknown–unknown' if parsing fails.
        """
        try:
            start_date = dates[0]
            end_date = dates[1]

            # Convert to datetime if they're strings
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date, errors="coerce")
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date, errors="coerce")

            # Format dates
            start_str = (
                start_date.strftime("%Y-%m") if pd.notna(start_date) else "unknown"
            )
            end_str = end_date.strftime("%Y-%m") if pd.notna(end_date) else "unknown"

            return f"{start_str}–{end_str}"
        except Exception as e:
            print(f"Warning: Error formatting date range: {e}")
            return "unknown–unknown"

    def calc_embeddings(self, df: pd.DataFrame, embedding_file: str):
        """
        Compute sentence embeddings for each event's descriptive text.

        Combines 'disorder_type', 'notes', 'actor1', and 'actor2' to form a text string per event,
        encodes it using a pre-trained SentenceTransformer model, and saves embeddings to a file.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with event-level data containing text fields.
        embedding_file : str
            File path to save computed embeddings.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with 'text_for_embedding' column added.
        """
        df["text_for_embedding"] = (
            df["disorder_type"].fillna("")
            + ". "
            + df["notes"].fillna("")
            + " Actors: "
            + df["actor1"].fillna("")
            + " vs "
            + df["actor2"].fillna("")
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        model = SentenceTransformer("all-mpnet-base-v2", device=device)

        # Generate embeddings
        embeddings = model.encode(
            df["text_for_embedding"].tolist(),
            batch_size=512,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=device,
            num_workers=4 if device == "cuda" else 1,
        )

        embeddings_cpu = embeddings.cpu().numpy()

        embeddings_path = DATA_DIR_PROCESSED / embedding_file
        np.save(embeddings_path, embeddings_cpu)
        print(f"Embeddings saved to {embeddings_path}")

        self.text_embeddings = embeddings_cpu
        return df

    def plot_umap_clusters(
        self, embeddings: np.ndarray = None, title: str = "UMAP Projection of Clusters"
    ):
        """
        Plot a 2D UMAP projection of the sequence embeddings, colored by cluster labels.

        Parameters
        ----------
        embeddings : np.ndarray, optional
            Embeddings to project. If None, uses stored text embeddings.
        title : str, optional
            Title of the plot.

        Returns
        -------
        np.ndarray
            UMAP 2D coordinates of the embeddings.
        """
        if embeddings is None and hasattr(self, "text_embeddings"):
            embeddings = self.text_embeddings
        elif embeddings is None:
            print(
                "No embeddings available. Run calc_embeddings first or provide embeddings."
            )
            return

        # Reduce dimensionality if needed
        if embeddings.shape[1] > 50:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=128)
            embeddings = pca.fit_transform(embeddings)

        # Perform UMAP reduction
        reducer = UMAP(n_neighbors=15, min_dist=0.1, init="pca", low_memory=True)
        X_proj = reducer.fit_transform(embeddings)

        # Get cluster labels
        labels = self.df["cluster_label"].values

        plt.figure(figsize=(12, 8))

        # Plot noise points
        is_noise = labels == -1
        if np.any(is_noise):
            plt.scatter(
                X_proj[is_noise, 0],
                X_proj[is_noise, 1],
                c="lightgray",
                s=1,
                alpha=0.3,
                label="Noise",
            )

        # Plot clustered points
        non_noise_mask = ~is_noise
        if np.any(non_noise_mask):
            scatter = plt.scatter(
                X_proj[non_noise_mask, 0],
                X_proj[non_noise_mask, 1],
                c=labels[non_noise_mask],
                cmap="tab20",
                s=4,
                alpha=0.7,
            )
            plt.colorbar(scatter, label="Cluster ID")

        plt.title(title)
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return X_proj

    def plot_top_clusters_umap(self, embeddings: np.ndarray = None, top_n: int = 20):
        """
        Plot a UMAP projection showing only the top N largest clusters by size.

        Parameters
        ----------
        embeddings : np.ndarray, optional
            Embeddings to project. If None, uses stored text embeddings.
        top_n : int, optional
            Number of largest clusters to display (default: 20).
        """
        if embeddings is None and hasattr(self, "text_embeddings"):
            embeddings = self.text_embeddings
        elif embeddings is None:
            print("No embeddings available.")
            return

        # Get top clusters by size
        cluster_sizes = self.df["cluster_label"].value_counts()
        top_clusters = cluster_sizes[cluster_sizes.index != -1].head(top_n).index

        # Filter data
        mask = self.df["cluster_label"].isin(top_clusters)
        X_proj = UMAP(n_neighbors=15, min_dist=0.1).fit_transform(embeddings[mask])
        labels_top = self.df.loc[mask, "cluster_label"]

        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x=X_proj[:, 0],
            y=X_proj[:, 1],
            hue=labels_top,
            palette="husl",
            s=4,
            linewidth=0,
            alpha=0.9,
            legend="full",
        )
        plt.title(f"UMAP – Top {top_n} Clusters (by size)")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.grid(True, alpha=0.3)
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)
        plt.tight_layout()
        plt.show()
