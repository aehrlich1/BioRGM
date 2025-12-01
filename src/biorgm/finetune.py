import argparse
import warnings
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torchinfo import summary

import wandb
from biorgm.model import CategoricalEncodingModel, FinetuneModel, PretrainModel
from biorgm.pretrain import load_pretrained_model
from biorgm.utils import PerformanceTracker, generate_random_alphanumeric, save_dict_to_csv


def create_pretrain_model_for_finetune(
    pretrain_model_name: str | None,
    dim_h: int,
    device: str,
) -> tuple[PretrainModel, int]:
    if pretrain_model_name is None:
        # Create random initialized model
        encoder = CategoricalEncodingModel().to(device)
        model = PretrainModel(encoder=encoder, dim_h=dim_h, dropout=0.1).to(device)
        return model, dim_h
    else:
        # Load pretrained model
        model, pretrain_config = load_pretrained_model(
            model_name=pretrain_model_name, device=device
        )
        return model, pretrain_config["dim_h"]


def create_finetune_model(
    pretrain_model: PretrainModel,
    dim_h: int,
    num_output_tasks: int,
    freeze_pretrain: bool,
    device: str,
) -> FinetuneModel:
    """Create finetune model with projection head."""

    model = FinetuneModel(
        pretrain_model=pretrain_model,
        in_dim=dim_h,
        out_dim=num_output_tasks,
    ).to(device)

    if freeze_pretrain:
        model.pretrain_model.freeze()

    return model


def create_molecule_net_dataset(dataset_name: str, data_dir: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = MoleculeNet(root=str(data_dir / "molecule_net"), name=dataset_name)
    return dataset


def create_train_val_test_split(dataset, train_ratio: float = 0.8, val_ratio: float = 0.1):
    test_ratio = 1.0 - train_ratio - val_ratio
    return random_split(dataset=dataset, lengths=[train_ratio, val_ratio, test_ratio])


def create_finetune_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def create_finetune_optimizer(model: torch.nn.Module, lr: float) -> Adam:
    return Adam(model.parameters(), lr=lr)


def create_finetune_loss_fn() -> BCELoss:
    return BCELoss()


def train_epoch(
    model: FinetuneModel,
    dataloader: DataLoader,
    optimizer: Adam,
    loss_fn: BCELoss,
    device: str,
) -> tuple[float, float]:
    model.train()
    batch_loss = 0
    y_true, y_pred = [], []

    for data in dataloader:
        data = data.to(device)
        out = model(data)
        loss = loss_fn(out, data.y)
        batch_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        y_true.append(data.y)
        y_pred.append(out.detach())

    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()

    average_loss = batch_loss / len(dataloader)
    roc_auc = roc_auc_score(y_true, y_pred)

    return average_loss, roc_auc


def validate_epoch(
    model: FinetuneModel,
    dataloader: DataLoader,
    loss_fn: BCELoss,
    device: str,
) -> tuple[float, float]:
    model.eval()
    batch_loss = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out = model(data)
            loss = loss_fn(out, data.y)
            batch_loss += loss.item()
            y_true.append(data.y)
            y_pred.append(out)

    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()

    average_loss = batch_loss / len(dataloader)
    roc_auc = roc_auc_score(y_true, y_pred)

    return average_loss, roc_auc


def test_epoch(
    model: FinetuneModel,
    dataloader: DataLoader,
    loss_fn: BCELoss,
    device: str,
) -> tuple[float, float]:
    model.eval()
    batch_loss = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out = model(data)
            loss = loss_fn(out, data.y)
            batch_loss += loss.item()
            y_true.append(data.y)
            y_pred.append(out)

    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()

    average_loss = batch_loss / len(dataloader)
    roc_auc = roc_auc_score(y_true, y_pred)

    return average_loss, roc_auc


class FinetuneTrainer:
    def __init__(self, config: dict, data_dir: Path, performance_tracker: PerformanceTracker):
        self.config = config
        self.data_dir = data_dir
        self.performance_tracker = performance_tracker
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {self.device}")

        # Load dataset and create splits
        dataset = create_molecule_net_dataset(config["dataset"], data_dir)
        train_dataset, val_dataset, test_dataset = create_train_val_test_split(dataset)

        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_finetune_dataloaders(
            train_dataset, val_dataset, test_dataset, config["batch_size"]
        )

        # Get number of output tasks from dataset
        num_output_tasks = torch.numel(dataset[0].y)

        # Create or load pretrain model
        pretrain_model, dim_h = create_pretrain_model_for_finetune(
            pretrain_model_name=config["pretrain_model"],
            dim_h=config["dim_h"],
            device=self.device,
        )
        # Update config with actual dim_h (might differ if loaded from checkpoint)
        self.config["dim_h"] = dim_h

        # Create finetune model
        self.model = create_finetune_model(
            pretrain_model=pretrain_model,
            dim_h=dim_h,
            num_output_tasks=num_output_tasks,
            freeze_pretrain=config["freeze_pretrain"],
            device=self.device,
        )

        # Create optimizer and loss function
        self.optimizer = create_finetune_optimizer(self.model, config["lr"])
        self.loss_fn = create_finetune_loss_fn()

        # Print model summary
        summary(self.model)

    def train(self) -> dict:
        for epoch in range(self.config["epochs"]):
            print(f"\nEpoch {epoch + 1}\n-------------------------------\n")
            self.performance_tracker.log({"epoch": epoch})

            # Training
            train_loss, train_roc = train_epoch(
                self.model, self.train_loader, self.optimizer, self.loss_fn, self.device
            )
            self.performance_tracker.log({"train_loss": train_loss})
            self.performance_tracker.log({"train_roc_auc": train_roc})

            # Validation
            val_loss, val_roc = validate_epoch(
                self.model, self.val_loader, self.loss_fn, self.device
            )
            self.performance_tracker.log({"valid_loss": val_loss})
            self.performance_tracker.log({"valid_roc_auc": val_roc})
            print(f"ROC AUC Valid: {val_roc}")

            # Test
            test_loss, test_roc = test_epoch(
                self.model, self.test_loader, self.loss_fn, self.device
            )
            self.performance_tracker.log({"test_loss": test_loss})
            self.performance_tracker.log({"test_roc_auc": test_roc})
            print(f"ROC AUC Test: {test_roc}")

            # Early stopping check
            self.performance_tracker.update_early_loss_state()
            if self.performance_tracker.early_stop:
                break

        # Save performance metrics
        self.performance_tracker.save_performance()
        return self.performance_tracker.get_results()


def main():
    parser = argparse.ArgumentParser(
        description="Finetune molecular GNN on property prediction tasks"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--dim_h", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze_pretrain", type=bool, default=True)
    parser.add_argument(
        "--pretrain_model",
        type=str,
        default=None,
        help="Name of pretrained model checkpoint, or None for random init",
    )

    parser.add_argument("--dataset", type=str, default="BACE", help="Name of MoleculeNet dataset")

    # Multiple runs (currently only runs=1 is used)
    parser.add_argument("--runs", type=int, default=3)

    args = parser.parse_args()
    config = vars(args)

    # Setup directories and tracking
    data_dir = Path("data")
    tracking_dir = Path("checkpoints/finetuned") / generate_random_alphanumeric(8)
    tracking_dir.mkdir(parents=True, exist_ok=True)

    # Create performance tracker
    performance_tracker = PerformanceTracker(tracking_dir=tracking_dir, id_run="1")

    # Train
    trainer = FinetuneTrainer(config, data_dir, performance_tracker)
    results = trainer.train()

    # Save final results
    final_results = config | results
    save_dict_to_csv([final_results], tracking_dir / "results.csv")

    print(f"\nTraining complete! Results saved to: {tracking_dir}")


if __name__ == "__main__":
    main()
