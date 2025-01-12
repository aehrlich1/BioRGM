import os
import sys

import argparse
import wandb
import torch
from pathlib import Path
from pytorch_metric_learning import losses, miners, samplers
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance, BaseDistance
from torch_geometric.loader import DataLoader

from src.data.data import PubchemDataset
from src.models.model import PretrainModel, CategoricalEncodingModel, OneHotEncoderModel
from src.utils import Checkpoint, read_config_file


class Pretrain:
    def __init__(self, params: dict = None, data_dir=None):
        self.params = params
        self.data_dir = data_dir
        self.dataset = None
        self.dataloader = None
        self.device = self._initialize_device()
        self.encoder_model = None
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
        self._initialize_encoder_model()
        self._initialize_model()
        self._initialize_loss_fn()
        self._initialize_optimizer()
        self._initialize_mining_fn()
        self._initialize_checkpoint()

    def load_pretrained_model(self, model_name) -> None:
        # TODO: epoch should be a parameter
        weights_file_path = Path(self.data_dir) / "models" / model_name / "epoch_5.pth"
        config_file_path = (
            Path(self.data_dir) / "models" / model_name / "config_pretrain.yml"
        )
        params: dict = read_config_file(config_file_path)

        encoder_model = self._get_encoder_model(params["encoder"])
        self.model = PretrainModel(encoder_model, params["dim_h"], params["dropout"])
        self.model.load_state_dict(torch.load(weights_file_path))

    def load_random_model(self, encoder_model, dim_h, dropout) -> None:
        self.model = PretrainModel(encoder_model, dim_h, dropout).to(self.device)

    def evaluate_model(self, datasets: list) -> None:
        pass

    def _initialize_wandb(self):
        wandb.init(project="BioRGM", config=self.params, mode="online")

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

    def _initialize_device(self) -> str:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        return device

    def _initialize_encoder_model(self) -> None:
        self.encoder_model = self._get_encoder_model(self.params["encoder"]).to(self.device)

    def _initialize_model(self) -> None:
        self.model = PretrainModel(
            encoder=self.encoder_model,
            dim_h=self.params["dim_h"],
            dropout=self.params["dropout"],
        ).to(self.device)

    def _initialize_loss_fn(self) -> None:
        self.loss_fn = losses.TripletMarginLoss(
            margin=self.params["margin"],
            distance=self._get_distance_metric(),
        )

    def _initialize_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.params["lr"],
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
        output_dir_name = wandb.run.name
        self.checkpoint = Checkpoint(self.data_dir, self.params, output_dir_name)

    def _get_encoder_model(self, encoder_name):
        if encoder_name == "embedding":
            return CategoricalEncodingModel()
        elif encoder_name == "one_hot":
            return OneHotEncoderModel()
        else:
            raise NotImplementedError

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
            label, data = data.y.to(self.device), data.to(self.device)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model with specific hyperparameters."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=32,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Absolute path to Root directory of dataset.",
    )
    parser.add_argument(
        "--dim_h",
        type=int,
        required=False,
        default=128,
        help="Hidden dimensions size.",
    )
    parser.add_argument(
        "--distance_metric",
        type=str,
        required=False,
        default="euclidean",
        help="Distance metric.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        required=False,
        default=0.1,
        help="Dropout ratio.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        required=False,
        default="embedding",
        help="Which encoder to use",
        choices=["embedding", "one_hot"],
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=5,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        required=True,
        default=5,
        help="File name (including extension) of the triplet dataset.",
    )
    parser.add_argument(
        "--finetune",
        type=bool,
        required=False,
        default=False,
        help="Whether to finetune and evaluate the model after each epoch.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        default="1e-6",
        help="Learning rate.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        required=False,
        default="0.1",
        help="Margin.",
    )
    parser.add_argument(
        "--num_samples_per_class",
        type=int,
        required=False,
        default="4",
        help="Number of samples per class.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default="0",
        help="Number of workers.",
    )
    parser.add_argument(
        "--type_of_triplets",
        type=str,
        required=False,
        default="all",
        help="Type of triplets.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        required=False,
        default="5e-4",
        help="Weight decay.",
    )

    args = parser.parse_args()
    args_dict = vars(args)

    pretrain = Pretrain(params=args_dict, data_dir=args_dict["data_dir"])
    pretrain.initialize_for_training()
    pretrain.train()
