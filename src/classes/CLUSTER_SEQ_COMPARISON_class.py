import warnings
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (adjusted_rand_score, fowlkes_mallows_score,
                             normalized_mutual_info_score)

warnings.filterwarnings("ignore")


class CLUSTER_SEQ_COMPARISON:
    """
    A class to compare multiple sequence clustering results.

    This class allows adding different clustering results (from various methods),
    calculating similarity metrics between them, analyzing sequence overlaps,
    and plotting comparison statistics. Metrics include Adjusted Rand Index (ARI),
    Normalized Mutual Information (NMI), and Fowlkes-Mallows Index (FMI).
    """

    def __init__(self):
        self.clustering_results = {}

    def add_clustering_method(self, method_name: str, cluster_object):
        """
        Add a clustering method and its sequence mapping to the comparison.

        Parameters
        ----------
        method_name : str
            Name of the clustering method.
        cluster_object : object
            Object of a clustering class instance that has an 'event_seq_mapping' attribute.

        Notes
        -----
        Noise events (labeled 'no_cluster' or -1) are ignored in the mappings.
        """
        event_seq_mapping = cluster_object.event_seq_mapping

        event_to_seq = {}
        seq_to_events = {}

        for event_id, seq_id in event_seq_mapping.items():
            if seq_id == "no_cluster" or seq_id == -1:
                continue

            event_id_str = str(event_id)
            event_to_seq[event_id_str] = seq_id

            if seq_id not in seq_to_events:
                seq_to_events[seq_id] = []
            seq_to_events[seq_id].append(event_id_str)

        self.clustering_results[method_name] = {
            "cluster_object": cluster_object,
            "event_seq_mapping": seq_to_events,  # seq_id -> [event_ids]
            "event_to_seq": event_to_seq,  # event_id -> seq_id
            "n_sequences": len(seq_to_events),
            "n_events": len(event_to_seq),
        }

        print(
            f"Added {method_name}: {len(seq_to_events)} sequences, {len(event_to_seq)} events"
        )

    def calculate_similarity_metrics(self) -> pd.DataFrame:
        """
        Compute similarity metrics between all added clustering methods.

        Returns
        -------
        pd.DataFrame
            DataFrame with pairwise comparisons, including:
            - Adjusted_Rand_Index
            - Normalized_Mutual_Info
            - Fowlkes_Mallows_Index
            - Number of sequences in each method
        """
        all_events = set()
        for method_data in self.clustering_results.values():
            all_events.update(method_data["event_to_seq"].keys())

        all_events = sorted(list(all_events))

        # Create comparison matrix where each row is an event, each column is a method
        comparison_data = {}
        for method_name, method_data in self.clustering_results.items():
            event_to_seq = method_data["event_to_seq"]
            comparison_data[method_name] = [
                event_to_seq.get(event_id, "no_cluster") for event_id in all_events
            ]

        comparison_df = pd.DataFrame(comparison_data, index=all_events)
        methods = list(self.clustering_results.keys())
        metrics_data = []

        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i <= j:
                    # Get labels for both methods FOR THE SAME EVENTS
                    labels1 = comparison_df[method1].values
                    labels2 = comparison_df[method2].values

                    # Convert to numeric labels for sklearn metrics
                    unique_labels1 = list(set(labels1))
                    unique_labels2 = list(set(labels2))

                    numeric_labels1 = [unique_labels1.index(label) for label in labels1]
                    numeric_labels2 = [unique_labels2.index(label) for label in labels2]

                    ari = adjusted_rand_score(numeric_labels1, numeric_labels2)
                    nmi = normalized_mutual_info_score(numeric_labels1, numeric_labels2)
                    fmi = fowlkes_mallows_score(numeric_labels1, numeric_labels2)

                    metrics_data.append(
                        {
                            "Method_1": method1,
                            "Method_2": method2,
                            "Adjusted_Rand_Index": ari,
                            "Normalized_Mutual_Info": nmi,
                            "Fowlkes_Mallows_Index": fmi,
                            "N_Sequences_1": len(unique_labels1),
                            "N_Sequences_2": len(unique_labels2),
                        }
                    )
        return pd.DataFrame(metrics_data)

    def plot_similarity_heatmap(self, metric="Normalized_Mutual_Info"):
        """
        Plot a heatmap of similarity between clustering methods.

        Parameters
        ----------
        metric : str, optional
            Metric to visualize ('Adjusted_Rand_Index', 'Normalized_Mutual_Info', 'Fowlkes_Mallows_Index').
            Default is 'Adjusted_Rand_Index'.

        Returns
        -------
        np.ndarray
            The symmetric similarity matrix used for plotting.
        """
        metrics_df = self.calculate_similarity_metrics()
        methods = list(self.clustering_results.keys())

        similarity_matrix = np.zeros((len(methods), len(methods)))

        for _, row in metrics_df.iterrows():
            i = methods.index(row["Method_1"])
            j = methods.index(row["Method_2"])
            similarity_matrix[i, j] = row[metric]
            similarity_matrix[j, i] = row[metric]  # Make symmetric

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            similarity_matrix,
            xticklabels=methods,
            yticklabels=methods,
            annot=True,
            cmap="viridis",
            vmin=0,
            vmax=1,
            square=True,
        )
        plt.title(f"Clustering Similarity: {metric}")
        plt.tight_layout()
        plt.show()

        return similarity_matrix

    def analyze_sequence_overlaps(self) -> pd.DataFrame:
        """
        Analyze event overlaps between sequences of different clustering methods.

        Returns
        -------
        pd.DataFrame
            Contains:
            - Method_1, Sequence_1, N_Events_1
            - Method_2, Sequence_2, N_Events_2
            - Overlap_Events
            - Jaccard_Index
            Sorted by descending Jaccard_Index.
        """
        overlap_data = []
        methods = list(self.clustering_results.keys())

        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i < j:  # Only compare each pair once

                    mapping1 = self.clustering_results[method1]["event_seq_mapping"]
                    mapping2 = self.clustering_results[method2]["event_seq_mapping"]

                    # For each sequence in method1, find best matching sequence in method2
                    for seq1_id, events1 in mapping1.items():
                        events1_set = set(events1)

                        best_overlap = 0
                        best_seq2 = None

                        for seq2_id, events2 in mapping2.items():
                            events2_set = set(events2)
                            overlap = len(events1_set & events2_set)

                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_seq2 = seq2_id

                        # Only add if there's actually an overlap
                        if best_overlap > 0 and best_seq2 is not None:
                            # Calculate Jaccard index
                            jaccard = best_overlap / len(
                                events1_set | set(mapping2[best_seq2])
                            )

                            overlap_data.append(
                                {
                                    "Method_1": method1,
                                    "Sequence_1": seq1_id,
                                    "N_Events_1": len(events1),
                                    "Method_2": method2,
                                    "Sequence_2": best_seq2,
                                    "N_Events_2": len(mapping2[best_seq2]),
                                    "Overlap_Events": best_overlap,
                                    "Jaccard_Index": jaccard,
                                }
                            )

        df_result = pd.DataFrame(overlap_data)
        if len(df_result) > 0:
            return df_result.sort_values("Jaccard_Index", ascending=False)
        else:
            return df_result

    def plot_sequence_size_comparison(self):
        """
        Plot various visual summaries of sequence size distributions across methods.

        Produces four plots:
        1. Histogram of sequence sizes (log scale)
        2. Number of sequences per method
        3. Total events clustered per method
        4. Average sequence size per method
        """
        _, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        ax = axes[0]
        for method_name, method_data in self.clustering_results.items():
            sizes = [
                len(events) for events in method_data["event_seq_mapping"].values()
            ]
            ax.hist(sizes, alpha=0.6, label=method_name, bins=20)
        ax.set_xlabel("Sequence Size (Number of Events)")
        ax.set_ylabel("Frequency (log scale)")
        ax.set_yscale("log")
        ax.set_title("Distribution of Sequence Sizes")
        ax.legend()

        ax = axes[1]
        methods = list(self.clustering_results.keys())
        n_sequences = [self.clustering_results[m]["n_sequences"] for m in methods]
        ax.bar(methods, n_sequences)
        ax.set_ylabel("Number of Sequences")
        ax.set_title("Total Number of Sequences per Method")
        plt.setp(ax.get_xticklabels(), rotation=45)

        ax = axes[2]
        n_events = [self.clustering_results[m]["n_events"] for m in methods]
        ax.bar(methods, n_events)
        ax.set_ylabel("Number of Events Clustered")
        ax.set_title("Total Events Clustered per Method")
        plt.setp(ax.get_xticklabels(), rotation=45)

        ax = axes[3]
        avg_sizes = []
        for method_name, method_data in self.clustering_results.items():
            sizes = [
                len(events) for events in method_data["event_seq_mapping"].values()
            ]
            avg_sizes.append(np.mean(sizes))

        ax.bar(methods, avg_sizes)
        ax.set_ylabel("Average Sequence Size")
        ax.set_title("Average Sequence Size per Method")
        plt.setp(ax.get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()

    def validate_event_coverage(self) -> Dict[str, Any]:
        """
        Validate coverage of events across all clustering methods.

        Returns
        -------
        dict
            Contains:
            - 'event_counts': number of events per method
            - 'common_events': number of events shared across all methods
            - 'common_ratio': proportion of shared events relative to the largest method
            - 'method_only_events': events unique to each method
            - 'noise_statistics': noise counts and ratios per method
            - 'coverage_warning': True if common coverage < 90% of largest method
        """
        method_events = {}
        for method_name, method_data in self.clustering_results.items():
            cluster_obj = method_data["cluster_object"]
            method_events[method_name] = set(cluster_obj.event_seq_mapping.index)

        all_methods = list(method_events.keys())
        common_events = set.intersection(*method_events.values())

        noise_stats = {}
        for method_name, method_data in self.clustering_results.items():
            cluster_obj = method_data["cluster_object"]
            total_events = len(cluster_obj.event_seq_mapping)
            noise_events = len(
                cluster_obj.event_seq_mapping[
                    cluster_obj.event_seq_mapping == "no_cluster"
                ]
            )
            noise_stats[method_name] = {
                "total": total_events,
                "noise": noise_events,
                "noise_ratio": noise_events / total_events if total_events > 0 else 0,
            }

        method_only_events = {}
        for method_name in all_methods:
            other_events = set()
            for other_method in all_methods:
                if other_method != method_name:
                    other_events.update(method_events[other_method])
            method_only_events[method_name] = method_events[method_name] - other_events

        return {
            "event_counts": {
                method: len(events) for method, events in method_events.items()
            },
            "common_events": len(common_events),
            "common_ratio": (
                len(common_events)
                / max(len(events) for events in method_events.values())
                if method_events
                else 0
            ),
            "method_only_events": {
                method: len(events) for method, events in method_only_events.items()
            },
            "noise_statistics": noise_stats,
            "coverage_warning": len(common_events)
            < max(len(events) for events in method_events.values()) * 0.9,
        }

    def print_summary(self) -> Dict[str, Any]:
        """
        Print a comprehensive summary of clustering comparison.

        Includes:
        - Noise ratios per method
        - Number of sequences, events, and average sequence size
        - Pairwise similarity scores (ARI, NMI)
        - Matrices of similarity metrics
        - Optional sequence overlap analysis

        Returns
        -------
        dict
            Dictionary containing:
            - 'validation': output of validate_event_coverage()
            - 'similarity_metrics': pairwise metrics DataFrame
            - 'sequence_overlaps': sequence overlaps DataFrame
        """
        # Validation
        validation = self.validate_event_coverage()

        if "warning" not in validation:
            print("\n" + "=" * 40)
            print("CLUSTERING VALIDATION SUMMARY")
            print("=" * 40)

            print("\nNoise ratios:")
            for method, stats in validation["noise_statistics"].items():
                ratio = stats["noise_ratio"]
                print(
                    f"  {method}: {ratio:.1%} ({stats['noise']:,}/{stats['total']:,})"
                )

        print("\n" + "=" * 40)
        print("CLUSTERING COMPARISON SUMMARY")
        print("=" * 40)

        # Calculate similarity metrics
        metrics_df = self.calculate_similarity_metrics()

        # Basic statistics
        print("\nMethod Statistics:")

        for method_name, method_data in self.clustering_results.items():
            sizes = [
                len(events) for events in method_data["event_seq_mapping"].values()
            ]
            avg_size = np.mean(sizes)
            print(
                f"  {method_name}: {method_data['n_sequences']} sequences, {method_data['n_events']} events (avg size: {avg_size:.1f})"
            )

        # Similarity summary
        non_diagonal = metrics_df[metrics_df["Method_1"] != metrics_df["Method_2"]]
        if len(non_diagonal) > 0:
            best_pair_ari = non_diagonal.loc[
                non_diagonal["Adjusted_Rand_Index"].idxmax()
            ]
            worst_pair_ari = non_diagonal.loc[
                non_diagonal["Adjusted_Rand_Index"].idxmin()
            ]
            best_pair_nmi = non_diagonal.loc[
                non_diagonal["Normalized_Mutual_Info"].idxmax()
            ]
            worst_pair_nmi = non_diagonal.loc[
                non_diagonal["Normalized_Mutual_Info"].idxmin()
            ]

            print(f"\nOverall Similarity Scores:")
            print(f"  Average ARI: {non_diagonal['Adjusted_Rand_Index'].mean():.4f}")
            print(f"  Average NMI: {non_diagonal['Normalized_Mutual_Info'].mean():.4f}")

            print(f"\nAdjusted Rand Index (ARI):")
            print(
                f"  Most similar: {best_pair_ari['Method_1']} & {best_pair_ari['Method_2']} (ARI: {best_pair_ari['Adjusted_Rand_Index']:.4f})"
            )
            print(
                f"  Least similar: {worst_pair_ari['Method_1']} & {worst_pair_ari['Method_2']} (ARI: {worst_pair_ari['Adjusted_Rand_Index']:.4f})"
            )

            print(f"\nNormalized Mutual Information (NMI):")
            print(
                f"  Most similar: {best_pair_nmi['Method_1']} & {best_pair_nmi['Method_2']} (NMI: {best_pair_nmi['Normalized_Mutual_Info']:.4f})"
            )
            print(
                f"  Least similar: {worst_pair_nmi['Method_1']} & {worst_pair_nmi['Method_2']} (NMI: {worst_pair_nmi['Normalized_Mutual_Info']:.4f})"
            )

        # Print similarity matrices
        self._print_similarity_matrices(metrics_df)

        overlaps_df = self.analyze_sequence_overlaps()

        return {
            "validation": validation,
            "similarity_metrics": metrics_df,
            "sequence_overlaps": overlaps_df,
        }

    def _print_similarity_matrices(self, metrics_df: pd.DataFrame):
        """
        Internal method to print ARI and NMI similarity matrices in a readable format.

        Parameters
        ----------
        metrics_df : pd.DataFrame
            DataFrame returned by calculate_similarity_metrics() containing pairwise metrics.
        """
        methods = list(self.clustering_results.keys())

        # Create matrices for both ARI and NMI
        ari_matrix = np.eye(len(methods))  # Initialize with 1s on diagonal
        nmi_matrix = np.eye(len(methods))  # Initialize with 1s on diagonal

        for _, row in metrics_df.iterrows():
            i = methods.index(row["Method_1"])
            j = methods.index(row["Method_2"])
            ari_matrix[i, j] = row["Adjusted_Rand_Index"]
            ari_matrix[j, i] = row["Adjusted_Rand_Index"]  # Make symmetric
            nmi_matrix[i, j] = row["Normalized_Mutual_Info"]
            nmi_matrix[j, i] = row["Normalized_Mutual_Info"]  # Make symmetric

        # Print ARI matrix
        print(f"\n" + "=" * 50)
        print("ADJUSTED RAND INDEX SIMILARITY MATRIX")
        print("=" * 50)

        # Header
        print(f"{'':12}", end="")
        for method in methods:
            print(f"{method:12}", end="")
        print()

        # Matrix rows
        for i, method in enumerate(methods):
            print(f"{method:12}", end="")
            for j in range(len(methods)):
                print(f"{ari_matrix[i, j]:12.4f}", end="")
            print()

        # Print NMI matrix
        print(f"\n" + "=" * 50)
        print("NORMALIZED MUTUAL INFORMATION SIMILARITY MATRIX")
        print("=" * 50)

        # Header
        print(f"{'':12}", end="")
        for method in methods:
            print(f"{method:12}", end="")
        print()

        # Matrix rows
        for i, method in enumerate(methods):
            print(f"{method:12}", end="")
            for j in range(len(methods)):
                print(f"{nmi_matrix[i, j]:12.4f}", end="")
            print()
