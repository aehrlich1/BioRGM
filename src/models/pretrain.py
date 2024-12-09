import os
import sys

import wandb
import torch
from pathlib import Path
from pytorch_metric_learning import losses, miners, samplers
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance, BaseDistance
from torch_geometric.loader import DataLoader

from src.data.data import PubchemDataset
from src.models.model import PretrainModel
from src.utils import Checkpoint, read_config_file


class Pretrain:
    def __init__(self, params: dict = None, data_dir=None):
        self.params = params
        self.data_dir = data_dir
        self.dataset = None
        self.dataloader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.sampler = None
        self.loss_fn = None
        self.mining_fn = None
        self.checkpoint = None

    def train(self) -> None:
        self.model.train()

        for epoch in range(self.params["epochs"]):
            print(f"\nEpoch {epoch}\n" + "-" * 30)
            self._train_loop()
            self.checkpoint.save(self.model, epoch)

        wandb.finish()

    def initialize_for_training(self) -> None:
        self._initialize_wandb()
        self._initialize_dataset()
        self._initialize_sampler()
        self._initialize_dataloader()
        self._initialize_model()
        self._initialize_loss_fn()
        self._initialize_optimizer()
        self._initialize_mining_fn()
        self._initialize_checkpoint()

    def load_pretrained_model(self, model_name) -> None:
        weights_file_path = Path(self.data_dir) / "models" / model_name / "epoch_2.pth"
        config_file_path = Path(self.data_dir) / "models" / model_name / "config_pretrain.yml"
        params: dict = read_config_file(config_file_path)

        self.model = PretrainModel(params["encoder"], params["dim_h"], params["dropout"])
        self.model.load_state_dict(torch.load(weights_file_path))

    def load_random_model(self, embedding_model, dim_h, dropout) -> None:
        self.model = PretrainModel(embedding_model, dim_h, dropout)

    def evaluate_model(self, datasets: list) -> None:
        # 1. Load dataset (EVAL)
        # 2. Pass the dataset through the pre-trained model
        # and obtain embeddings. batch_size=len(dataset)
        # 3. Append embeddings to the Data object
        # 4. Calculate cosine pairwise distances for embeddings
        # 5. Calculate euclidean distances of target properties
        # x. Calculate Improvement Rate S_IR
        # x. Calculate Average Deviation S_AD
        # return: a dictionary {"ESOL": {S_IR: 1.0, S_ID: 1.2}, "FreeSolv": {...}}
        pass

    def _initialize_wandb(self):
        wandb.init(project="BioRGM", config=self.params)

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

    def _initialize_model(self) -> None:
        self.model = PretrainModel(
            embedding_model=self.params["encoder"],
            dim_h=self.params["dim_h"],
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
        output_dir = wandb.run.name
        self.checkpoint = Checkpoint(self.data_dir, self.params, output_dir)

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
            embeddings = self.model(data)

            indices_tuple = self.mining_fn(embeddings, label)
            loss = self.loss_fn(embeddings, label, indices_tuple)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if i % 20 == 0:
                print(
                    f"Iteration {i}: Loss = {loss:.3g}, Mined triplets = {self.mining_fn.num_triplets}"
                )
                wandb.log({"Loss": loss, "Mined Triplets": self.mining_fn.num_triplets})
