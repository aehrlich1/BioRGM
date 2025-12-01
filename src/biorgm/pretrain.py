import argparse
from pathlib import Path

import torch
from pytorch_metric_learning import losses, miners, samplers
from pytorch_metric_learning.distances import BaseDistance, CosineSimilarity, LpDistance
from torch_geometric.loader import DataLoader

import wandb
from biorgm.data import PubchemDataset
from biorgm.model import CategoricalEncodingModel, OneHotEncoderModel, PretrainModel
from biorgm.utils import Checkpoint, read_config_file


def get_distance_metric(metric_name: str) -> BaseDistance:
    if metric_name == "euclidean":
        return LpDistance(normalize_embeddings=True, p=2, power=1)
    elif metric_name == "cosine":
        return CosineSimilarity()
    else:
        raise ValueError(f"Unknown distance metric: {metric_name}")


def create_encoder(encoder_type: str, device: str):
    if encoder_type == "embedding":
        return CategoricalEncodingModel().to(device)
    elif encoder_type == "one_hot":
        return OneHotEncoderModel().to(device)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def create_pretrain_model(config: dict, device: str) -> PretrainModel:
    encoder = create_encoder(config["encoder"], device)
    model = PretrainModel(encoder=encoder, dim_h=config["dim_h"], dropout=config["dropout"])
    return model.to(device)


def create_dataloader(dataset, batch_size: int, num_workers: int, sampler=None) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)


def create_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        params=model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )


def create_sampler(dataset, config: dict) -> samplers.MPerClassSampler:
    return samplers.MPerClassSampler(
        labels=dataset.y,
        m=config["num_samples_per_class"],
        batch_size=config["batch_size"],
        length_before_new_iter=len(dataset),
    )


def create_loss_fn(config: dict) -> losses.TripletMarginLoss:
    distance = get_distance_metric(config["distance_metric"])
    return losses.TripletMarginLoss(margin=config["margin"], distance=distance)


def create_mining_fn(config: dict) -> miners.TripletMarginMiner:
    distance = get_distance_metric(config["distance_metric"])
    return miners.TripletMarginMiner(
        margin=config["margin"],
        distance=distance,
        type_of_triplets=config["type_of_triplets"],
    )


def load_pretrained_model(
    model_name: str,
    device: str,
    epoch: int = 4,
) -> tuple[PretrainModel, dict]:
    checkpoint_dir = Path("checkpoints/pretrained") / model_name
    weights_path = checkpoint_dir / f"epoch_{epoch}.pth"
    config_path = checkpoint_dir / "config_pretrain.yml"

    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = read_config_file(config_path)
    model = create_pretrain_model(config, device)
    model.load_state_dict(torch.load(weights_path, weights_only=True))

    return model, config


def load_checkpoint_config(model_name: str) -> dict:
    config_path = Path("checkpoints/pretrained") / model_name / "config_pretrain.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return read_config_file(config_path)


class PretrainTrainer:
    def __init__(self, config: dict, dataset):
        self.config = config
        self.dataset = dataset
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.sampler = create_sampler(dataset, config)
        self.dataloader = create_dataloader(
            dataset,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=self.sampler,
        )
        self.model = create_pretrain_model(config, self.device)
        self.optimizer = create_optimizer(self.model, config)
        self.loss_fn = create_loss_fn(config)
        self.mining_fn = create_mining_fn(config)

        # Initialize tracking
        wandb.init(project="BioRGM", config=config, mode="online", reinit=True)
        self.checkpoint = Checkpoint(config, output_dir_name=wandb.run.name)

        print(f"Using device: {self.device}")

    def train(self) -> PretrainModel:
        self.model.train()

        for epoch in range(self.config["epochs"]):
            print(f"\nEpoch {epoch}\n" + "-" * 30)
            self._train_epoch(epoch)
            self.checkpoint.save(self.model, epoch)

        wandb.finish()
        return self.model

    def _train_epoch(self, epoch: int) -> None:
        for i, batch in enumerate(self.dataloader):
            # Move to device
            labels = batch.y.to(self.device)
            batch = batch.to(self.device)

            # Forward pass
            embeddings = self.model(batch)

            # Mine hard triplets and compute loss
            indices = self.mining_fn(embeddings, labels)
            loss = self.loss_fn(embeddings, labels, indices)

            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Logging
            if i % 20 == 0:
                print(
                    f"Iteration {i}: Loss = {loss:.3g}, "
                    f"Mined triplets = {self.mining_fn.num_triplets}"
                )
                wandb.log({"Loss": loss.item(), "Mined Triplets": self.mining_fn.num_triplets})


def main():
    parser = argparse.ArgumentParser(description="Pretrain molecular GNN with metric learning")
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--dim_h", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1.0e-6)
    parser.add_argument("--weight_decay", type=float, default=5.0e-4)
    parser.add_argument(
        "--encoder", type=str, default="embedding", choices=["embedding", "one_hot"]
    )
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument(
        "--distance_metric", type=str, default="cosine", choices=["cosine", "euclidean"]
    )
    parser.add_argument("--type_of_triplets", type=str, default="all")
    parser.add_argument("--num_samples_per_class", type=int, default=2)
    parser.add_argument("--file_name", type=str, default="pubchem_1k_triplets.csv")
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    config = vars(args)

    # Load dataset
    data_dir = Path("data")
    dataset = PubchemDataset(root=data_dir / "pubchem" / "processed", file_name=config["file_name"])

    # Train
    trainer = PretrainTrainer(config, dataset)
    trained_model = trainer.train()

    print(f"\nTraining complete! Model saved to: {trainer.checkpoint.output_dir_name}")


if __name__ == "__main__":
    main()
