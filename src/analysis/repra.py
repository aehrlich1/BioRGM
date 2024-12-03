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
            dataset_name: str,
            data_dir: str,
            task_type: str = "regression",
            metric: str = "min_max_eud",
    ):
        self.model = model
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.task_type = task_type
        self.metric = metric

    def analyze(self):
        dataset = self.get_dataset()
        dataloader = self.get_dataloader(dataset)
        embeddings = self.calculate_embeddings(self.model, dataloader)
        pairwise_embeddings_distances = self.calculate_pairwise_distances(embeddings)
        properties = self.get_properties(dataset)
        pairwise_properties_distances = self.calculate_pairwise_distances(properties)
        thresholds = self.calculate_thresholds(
            pairwise_embeddings_distances=pairwise_embeddings_distances,
            pairwise_properties_distances=pairwise_properties_distances,
        )
        c1_ptm, c4_ptm = self.calculate_c1_and_c4_count(
            pairwise_embeddings_distances=pairwise_embeddings_distances,
            pairwise_properties_distances=pairwise_properties_distances,
            thresholds=thresholds,
        )

        model_fp = self.get_model_fp()
        embeddings_fp = self.calculate_embeddings(model_fp, dataloader)
        pairwise_embeddings_distances_fp = self.calculate_pairwise_distances(
            embeddings_fp
        )
        c1_fp, c4_fp = self.calculate_c1_and_c4_count(
            pairwise_embeddings_distances=pairwise_embeddings_distances_fp,
            pairwise_properties_distances=pairwise_properties_distances,
            thresholds=thresholds,
        )

        improvement_rate = self.calculate_improvement_rate(c1_ptm, c4_ptm, c1_fp, c4_fp)
        average_deviation = self.calculate_average_deviation(
            pairwise_embeddings_distances=pairwise_embeddings_distances,
            pairwise_properties_distances=pairwise_properties_distances,
            thresholds=thresholds,
        )

        self.plot_rps_map(
            pairwise_embeddings_distances=pairwise_embeddings_distances,
            pairwise_properties_distances=pairwise_properties_distances,
            thresholds=thresholds,
        )

        return improvement_rate, average_deviation

    def get_dataset(self):
        return MoleculeNet(root=self.data_dir, name=self.dataset_name)

    @staticmethod
    def get_model_fp():
        return ExtendedConnectivityFingerprintModel()

    @staticmethod
    def get_dataloader(dataset):
        return DataLoader(dataset, batch_size=len(dataset), num_workers=0)

    @staticmethod
    def calculate_embeddings(model, dataloader):
        embeddings = model(next(iter(dataloader)))
        if torch.is_tensor(embeddings):
            embeddings = embeddings.detach().numpy()
        return embeddings

    @staticmethod
    def calculate_pairwise_distances(embeddings):
        return Repra.min_max_eud(embeddings)

    @staticmethod
    def calculate_c1_and_c4_count(
            pairwise_embeddings_distances, pairwise_properties_distances, thresholds
    ):
        pairwise_embeddings_similarities = 1 - pairwise_embeddings_distances

        c1 = np.sum(
            (pairwise_embeddings_similarities < 1 - thresholds["delta_2"])
            & (pairwise_properties_distances < thresholds["epsilon_2"])
        )

        c4 = np.sum(
            (pairwise_embeddings_similarities > 1 - thresholds["delta_1"])
            & (pairwise_properties_distances > thresholds["epsilon_1"])
        )

        return c1, c4

    @staticmethod
    def calculate_thresholds(
            pairwise_embeddings_distances, pairwise_properties_distances
    ):
        median = np.median(pairwise_embeddings_distances)
        d_near_mask = pairwise_embeddings_distances < median
        d_far_mask = ~d_near_mask

        delta_1 = np.mean(pairwise_embeddings_distances[d_near_mask])
        epsilon_1 = np.mean(pairwise_properties_distances[d_near_mask])

        delta_2 = np.mean(pairwise_embeddings_distances[d_far_mask])
        epsilon_2 = np.mean(pairwise_properties_distances[d_far_mask])

        return {
            "delta_1": delta_1,
            "epsilon_1": epsilon_1,
            "delta_2": delta_2,
            "epsilon_2": epsilon_2,
        }

    @staticmethod
    def get_properties(dataset):
        return dataset.y.detach().numpy().reshape(-1, 1)

    @staticmethod
    def min_max_eud(X):
        pairwise_dist = pdist(X, metric="euclidean")

        min_pairwise_dist = min(pairwise_dist)
        max_pairwise_dist = max(pairwise_dist)
        return (pairwise_dist - min_pairwise_dist) / (
                max_pairwise_dist - min_pairwise_dist
        )

    @staticmethod
    def calculate_improvement_rate(c1_ptm, c4_ptm, c1_fp, c4_fp):
        improvement_rate = (c1_ptm / c1_fp) + (c4_ptm / c4_fp)
        return improvement_rate

    @staticmethod
    def calculate_average_deviation(
            pairwise_embeddings_distances, pairwise_properties_distances, thresholds
    ):
        (
            pairwise_embeddings_similarities_r1,
            pairwise_properties_distances_r1,
            pairwise_embeddings_similarities_r4,
            pairwise_properties_distances_r4,
        ) = Repra.get_r1_r4_points(
            pairwise_embeddings_distances, pairwise_properties_distances, thresholds
        )

        abs_prop_dist_diff_r1 = np.abs(
            pairwise_properties_distances_r1 - thresholds["epsilon_2"]
        )
        abs_sim_dist_diff_r1 = np.abs(
            pairwise_embeddings_similarities_r1 - (1 - thresholds["delta_2"])
        )

        abs_prop_dist_diff_r4 = np.abs(
            pairwise_properties_distances_r4 - thresholds["epsilon_1"]
        )
        abs_sim_dist_diff_r4 = np.abs(
            pairwise_embeddings_similarities_r4 - (1 - thresholds["delta_1"])
        )

        m_r1 = np.sum(np.minimum(abs_prop_dist_diff_r1, abs_sim_dist_diff_r1))
        m_r4 = np.sum(np.minimum(abs_prop_dist_diff_r4, abs_sim_dist_diff_r4))
        s_ad = (m_r1 + m_r4) / (len(pairwise_embeddings_distances))

        return s_ad

    @staticmethod
    def get_r1_r4_points(
            pairwise_embeddings_distances, pairwise_properties_distances, thresholds
    ):
        pairwise_embeddings_similarities = 1 - pairwise_embeddings_distances

        r1_mask = (pairwise_embeddings_similarities < 1 - thresholds["delta_2"]) & (
                pairwise_properties_distances < thresholds["epsilon_2"]
        )

        r4_mask = (pairwise_embeddings_similarities > 1 - thresholds["delta_1"]) & (
                pairwise_properties_distances > thresholds["epsilon_1"]
        )

        return (
            pairwise_embeddings_similarities[r1_mask],
            pairwise_properties_distances[r1_mask],
            pairwise_embeddings_similarities[r4_mask],
            pairwise_properties_distances[r4_mask],
        )

    @staticmethod
    def plot_rps_map(
            pairwise_embeddings_distances,
            pairwise_properties_distances,
            thresholds,
    ):
        pairwise_embeddings_similarities = 1 - pairwise_embeddings_distances
        x = pairwise_embeddings_similarities
        y = pairwise_properties_distances

        fig, ax = plt.subplots(figsize=(8, 6), dpi=200, facecolor="w", edgecolor="k")
        ax.set_position([0.13, 0.13, 0.8, 0.8])
        ax.scatter(x, y, marker="o", s=12, color="salmon", edgecolors="grey")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        rectangle_sh = Rectangle(
            (0, 0),
            height=thresholds["epsilon_2"],
            width=1 - thresholds["delta_2"],
            linewidth=1,
            edgecolor="grey",
            facecolor="grey",
            alpha=0.4,
        )

        rectangle_ac = Rectangle(
            (1 - thresholds["delta_1"], thresholds["epsilon_1"]),
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
