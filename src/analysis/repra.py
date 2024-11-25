import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial.distance import pdist
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

from src.models.model import ExtendedConnectivityFingerprintModel


class Repra:
    def __init__(
        self,
        model,
        datasets: list,
        data_dir,
        task_type="regression",
        metric="min_max_eud",
    ):
        self.model = model
        self.datasets = datasets
        self.data_dir = data_dir
        self.task_type = task_type
        self.metric = metric
        self.dataset = None
        self.dataloader = None
        self.embeddings = None
        self.properties = None
        self.pairwise_embeddings_distances = None
        self.pairwise_embeddings_similarity = None
        self.pairwise_properties_similarity = None
        self.thresholds = None
        self.fingerprints = None
        self.c1_c4_counts = None
        self.c1_c4_ecfp_counts = None

        self.initialize_dataset(datasets[0])
        self.initialize_dataloader()
        self.initialize_properties()
        self.initialize_embeddings()
        self.initialize_pairwise_embeddings_distances()
        self.initialize_pairwise_embeddings_similarity()
        self.initialize_pairwise_property_similarity()
        self.initialize_thresholds()
        self.initialize_c1_c4_counts()
        self.initialize_c1_c4_ecfp_counts()
        self.calculate_improvement_rate()
        self.calculate_average_deviation()
        self.plot_rps_map()

    def initialize_dataset(self, name) -> None:
        self.dataset = MoleculeNet(root=self.data_dir, name=name)

    def initialize_dataloader(self) -> None:
        self.dataloader = DataLoader(
            self.dataset, batch_size=len(self.dataset), num_workers=0
        )

    def initialize_properties(self) -> None:
        self.properties = self.dataset.y.detach().numpy().reshape(-1, 1)

    def initialize_embeddings(self) -> None:
        self.embeddings = self.calculate_embeddings(self.model)

    def initialize_pairwise_embeddings_distances(self) -> None:
        self.pairwise_embeddings_distances = self.calculate_pairwise_distances(
            self.embeddings
        )

    def initialize_pairwise_embeddings_similarity(self) -> None:
        self.pairwise_embeddings_similarity = self.calculate_pairwise_similarity(
            self.embeddings
        )

    def calculate_embeddings(self, model):
        embeddings = model(next(iter(self.dataloader)))
        if torch.is_tensor(embeddings):
            embeddings = embeddings.detach().numpy()

        return embeddings

    def calculate_pairwise_distances(self, embeddings):
        return self.min_max_eud(embeddings)

    def calculate_pairwise_similarity(self, embeddings):
        # cos_sim = 1 - cos_dist
        # returns a sparse vector with all pairwise distances
        return 1 - self.min_max_eud(embeddings)

    def initialize_pairwise_property_similarity(self) -> None:
        self.pairwise_properties_similarity = self.min_max_eud(self.properties)

    def initialize_thresholds(self) -> None:
        median = np.median(self.pairwise_embeddings_distances)
        d_near_mask = self.pairwise_embeddings_distances < median
        d_far_mask = ~d_near_mask

        delta_1 = np.mean(self.pairwise_embeddings_distances[d_near_mask])
        epsilon_1 = np.mean(self.pairwise_properties_similarity[d_near_mask])

        delta_2 = np.mean(self.pairwise_embeddings_distances[d_far_mask])
        epsilon_2 = np.mean(self.pairwise_properties_similarity[d_far_mask])

        self.thresholds = {
            "delta_1": delta_1,
            "epsilon_1": epsilon_1,
            "delta_2": delta_2,
            "epsilon_2": epsilon_2,
        }

        print(self.thresholds)

    def initialize_c1_c4_counts(self) -> None:
        c1, c4 = self.calculate_c1_and_c4_count(self.pairwise_embeddings_similarity)
        self.c1_c4_counts = {
            "c1": c1,
            "c4": c4,
        }

    @staticmethod
    def min_max_eud(X):
        pairwise_dist = pdist(X, metric="euclidean")

        min_pairwise_dist = min(pairwise_dist)
        max_pairwise_dist = max(pairwise_dist)
        return (pairwise_dist - min_pairwise_dist) / (
            max_pairwise_dist - min_pairwise_dist
        )

    def initialize_c1_c4_ecfp_counts(self) -> None:
        ecfp_model = ExtendedConnectivityFingerprintModel()
        ecfp_embeddings = self.calculate_embeddings(ecfp_model)
        ecfp_pairwise_embedding_similarities = self.calculate_pairwise_similarity(
            ecfp_embeddings
        )
        c1_ecfp, c4_ecfp = self.calculate_c1_and_c4_count(
            ecfp_pairwise_embedding_similarities
        )
        self.c1_c4_ecfp_counts = {
            "c1": c1_ecfp,
            "c4": c4_ecfp,
        }

    def calculate_c1_and_c4_count(self, pairwise_embeddings_similarity):
        c1 = np.sum(
            (pairwise_embeddings_similarity < 1 - self.thresholds["delta_2"])
            & (self.pairwise_properties_similarity < self.thresholds["epsilon_2"])
        )

        c4 = np.sum(
            (pairwise_embeddings_similarity > 1 - self.thresholds["delta_1"])
            & (self.pairwise_properties_similarity > self.thresholds["epsilon_1"])
        )

        return c1, c4

    def calculate_improvement_rate(self):
        improvement_rate = (self.c1_c4_counts["c1"] / self.c1_c4_ecfp_counts["c1"]) + (
            self.c1_c4_counts["c4"] / self.c1_c4_ecfp_counts["c4"]
        )
        print(f"Improvement rate: {improvement_rate}")

    def calculate_average_deviation(self):
        abs_prop_dist_diff = np.abs(self.pairwise_properties_similarity - self.thresholds["epsilon_2"])
        abs_sim_dist_diff = np.abs(self.pairwise_embeddings_similarity - (1 - self.thresholds["delta_2"]))

        m = np.sum(np.minimum(abs_prop_dist_diff, abs_sim_dist_diff))
        s_ad = m / (len(abs_prop_dist_diff))
        print(s_ad)
        pass

    def plot_rps_map(self):
        x = self.pairwise_embeddings_similarity
        y = self.pairwise_properties_similarity

        fig, ax = plt.subplots(figsize=(8, 6), dpi=200, facecolor="w", edgecolor="k")
        ax.set_position([0.13, 0.13, 0.8, 0.8])
        ax.scatter(x, y, marker="o", s=12, color="salmon", edgecolors="grey")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        rectangle_sh = Rectangle(
            (0, 0),
            height=self.thresholds["epsilon_2"],
            width=1 - self.thresholds["delta_2"],
            linewidth=1,
            edgecolor="grey",
            facecolor="grey",
            alpha=0.4,
        )

        rectangle_ac = Rectangle(
            (1 - self.thresholds["delta_1"], self.thresholds["epsilon_1"]),
            height=1,
            width=1,
            linewidth=1,
            edgecolor="grey",
            facecolor="grey",
            alpha=0.4,
        )

        ax.add_patch(rectangle_sh)
        ax.add_patch(rectangle_ac)

        plt.xlabel("Representational Similarity")
        plt.ylabel("Property Distance")
        plt.show()
