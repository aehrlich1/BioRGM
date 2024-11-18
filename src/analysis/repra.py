import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader


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
        self.pairwise_embeddings_similarity = None
        self.pairwise_properties_similarity = None
        self.thresholds = None
        self.fingerprints = None

        self.initialize_dataset(datasets[0])
        self.initialize_dataloader()
        self.initialize_properties()
        self.calculate_embeddings()
        self.calculate_pairwise_similarity()
        self.calculate_pairwise_property_similarity()
        self.calculate_thresholds()

        self.plot_rps_map()
        # self.get_ac_and_sh_count()
        # self.calculate_fingerprints()
        # self.calculate_improvement_rate()

    def initialize_dataset(self, name) -> None:
        self.dataset = MoleculeNet(root=self.data_dir, name=name)

    def initialize_dataloader(self) -> None:
        self.dataloader = DataLoader(
            self.dataset, batch_size=len(self.dataset), num_workers=0
        )

    def initialize_properties(self):
        self.properties = self.dataset.y.detach().numpy().reshape(-1, 1)

    def calculate_embeddings(self):
        embeddings = self.model(next(iter(self.dataloader)))
        if torch.is_tensor(embeddings):
            self.embeddings = embeddings.detach().numpy()
        self.embeddings = embeddings

    def calculate_pairwise_similarity(self):
        # cos_sim = 1 - cos_dist
        # returns a sparse vector with all pairwise distances
        self.pairwise_embeddings_similarity = 1 - self.min_max_eud(self.embeddings)

    def calculate_pairwise_property_similarity(self):
        self.pairwise_properties_similarity = self.min_max_eud(self.properties)

    def calculate_thresholds(self):
        median = np.median(self.pairwise_embeddings_similarity)
        d_near_mask = self.pairwise_embeddings_similarity < median
        d_far_mask = ~d_near_mask

        delta_1 = np.mean(self.pairwise_embeddings_similarity[d_near_mask])
        epsilon_1 = np.mean(self.pairwise_properties_similarity[d_near_mask])

        delta_2 = np.mean(self.pairwise_embeddings_similarity[d_far_mask])
        epsilon_2 = np.mean(self.pairwise_properties_similarity[d_far_mask])

        self.thresholds = {
            "delta_1": delta_1,
            "epsilon_1": epsilon_1,
            "delta_2": delta_2,
            "epsilon_2": epsilon_2,
        }

    @staticmethod
    def min_max_eud(X):
        pairwise_dist = pdist(X, metric="euclidean")

        min_pairwise_dist = min(pairwise_dist)
        max_pairwise_dist = max(pairwise_dist)
        return (pairwise_dist - min_pairwise_dist) / (
            max_pairwise_dist - min_pairwise_dist
        )

    def calculate_fingerprint_embeddings(self):
        pass

    def get_ac_and_sh_count(self):
        # SH
        a = self.pairwise_embeddings_similarity < 1 - self.thresholds["delta_2"]
        if (self.pairwise_embeddings_similarity < 1 - self.thresholds["delta_1"]) & (
            self.pairwise_properties_similarity < self.properties["epsilon_1"]
        ):
            pass

    def calculate_average_deviation(self):
        pass

    def calculate_improvement_rate(self):
        pass

    def plot_rps_map(self):
        x = self.pairwise_embeddings_similarity
        y = self.pairwise_properties_similarity

        fig = plt.figure(figsize=(8, 6), dpi=200, facecolor="w", edgecolor="k")
        axes = fig.add_axes([0.13, 0.13, 0.8, 0.8])
        axes.scatter(x, y, marker="o", s=12, color="salmon", edgecolors="grey")
        plt.show()

    def _min_max_eud(self):
        pass
