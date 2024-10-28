import os
import sys
import torch

from pytorch_metric_learning import losses, miners, samplers
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance, BaseDistance
from torch_geometric.loader import DataLoader

from src.data.data import PubchemDataset
from src.models.model import GIN, CategoricalEmbeddingModel
from src.utils import Checkpoint


class Pretrain:
    def __init__(self, params: dict = None, data_dir=None):
        self.params = params
        self.data_dir = data_dir
        self.dataset = None
        self.dataloader = None
        self.node_feature_embeddings = None
        self.edge_feature_embeddings = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.sampler = None
        self.loss_fn = None
        self.mining_fn = None
        self.checkpoint = None

        self._initialize_dataset()
        self._initialize_sampler()
        self._initialize_dataloader()
        self._initialize_feature_embeddings()
        self._initialize_model()
        self._initialize_loss_fn()
        self._initialize_optimizer()
        self._initialize_mining_fn()
        self._initialize_checkpoint()

    def train(self) -> None:
        self.model.train()

        for epoch in range(self.params["epochs"]):
            print(f"\nEpoch {epoch}\n" + "-" * 30)
            self._train_loop()
            self.checkpoint.save(self.model, epoch)

    def _initialize_dataset(self) -> None:
        self.dataset = PubchemDataset(
            root=os.path.join(self.data_dir, "processed"),
            file_name=self.params["file_name"],
        )

    def _initialize_dataloader(self) -> None:
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.params["batch_size"],
            num_workers=self.params["num_workers"],
            sampler=self.sampler,
        )

    def _initialize_feature_embeddings(self) -> None:
        self.node_feature_embeddings = CategoricalEmbeddingModel(category_type="node")
        self.edge_feature_embeddings = CategoricalEmbeddingModel(category_type="edge")

    def _initialize_model(self) -> None:
        self.model = GIN(
            dim_h=self.params["dim_h"],
            dim_in=self.node_feature_embeddings.get_node_feature_dim(),
            edge_dim=self.edge_feature_embeddings.get_edge_feature_dim(),
            dropout=self.params["dropout"],
        )

    def _initialize_loss_fn(self) -> None:
        self.loss_fn = losses.TripletMarginLoss(
            margin=self.params["margin"],
            distance=self._get_distance_metric(),
        )

    def _initialize_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )

    def _initialize_sampler(self) -> None:
        self.sampler = samplers.MPerClassSampler(
            self.dataset.y,
            m=self.params["num_samples_per_class"],
            batch_size=self.params["batch_size"],
            length_before_new_iter=len(self.dataset),
        )

    def _initialize_mining_fn(self) -> None:
        self.mining_fn = miners.TripletMarginMiner(
            margin=self.params["margin"],
            distance=self._get_distance_metric(),
            type_of_triplets=self.params["type_of_triplets"],
        )

    def _initialize_checkpoint(self) -> None:
        self.checkpoint = Checkpoint(self.data_dir, self.params)

    def _get_distance_metric(self) -> BaseDistance:
        match self.params["distance_metric"]:
            case "euclidean":
                return LpDistance(normalize_embeddings=True, p=2, power=1)
            case "cosine":
                return CosineSimilarity()
            case _:
                print("Unknown distance metric.")
                sys.exit(1)

    def _train_loop(self) -> None:
        for i, data in enumerate(self.dataloader):
            label, data = data.y, data

            # use embedded feature vectors
            data.x = self.node_feature_embeddings(data.x)
            data.edge_attr = self.edge_feature_embeddings(data.edge_attr)

            # compute prediction and loss
            embeddings = self.model(data)

            indices_tuple = self.mining_fn(embeddings, label)
            loss = self.loss_fn(embeddings, label, indices_tuple)

            # backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if i % 20 == 0:
                print(
                    f"Iteration {i}: Loss = {loss:.3g}, Mined triplets = {self.mining_fn.num_triplets}"
                )
