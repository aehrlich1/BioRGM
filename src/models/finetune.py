import torch
import wandb
from dataclasses import dataclass, asdict
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
from torch.nn import BCELoss
from torch.utils.data import random_split
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

from src.models.model import FinetuneModel


@dataclass
class FinetuneParams:
    batch_size: int
    dataset: str
    epochs: int
    freeze_pretrain: bool
    lr: float
    classifier: str = "MLP, SVM, Linear, ..."
    pretrain_model: str = "Random"


def finetune(pretrain_model, params: FinetuneParams, data_dir: str) -> None:
    wandb.init(project="BioRGM_finetune", config=asdict(params), mode="offline")

    dataset = _get_dataset(params.dataset, data_dir)
    num_output_tasks = _get_num_output_tasks(dataset)
    train_dataloader, test_dataloader = _get_dataloaders(dataset, params.batch_size)
    finetune_model = _initialize_finetune_model(
        pretrain_model, num_output_tasks, params.freeze_pretrain
    )
    optimizer = Adam(finetune_model.parameters(), lr=params.lr)
    loss_fn = BCELoss()

    _test_loop(finetune_model, test_dataloader, loss_fn, epoch=0)
    for epoch in range(params.epochs):
        print(f"\nEpoch {epoch+1}\n-------------------------------\n")
        _train_loop(finetune_model, train_dataloader, optimizer, loss_fn, epoch)
        _test_loop(finetune_model, test_dataloader, loss_fn, epoch)


def _train_loop(model, dataloader, optimizer, loss_fn, epoch):
    model.train()
    batch_loss = 0

    for data in dataloader:
        out = model(data)
        loss = loss_fn(out, data.y)
        batch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = batch_loss / len(dataloader)
    wandb.log({"training loss": average_loss}, step=epoch)


def _test_loop(model, dataloader, loss_fn, epoch):
    model.eval()
    batch_loss = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for data in dataloader:
            out = model(data)
            loss = loss_fn(out, data.y)
            batch_loss += loss.item()
            y_true.append(data.y)
            y_pred.append(out)

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    average_loss = batch_loss / len(dataloader)
    roc_auc = roc_auc_score(y_true, y_pred)

    wandb.log({"test loss": average_loss}, step=epoch)
    wandb.log({"ROC AUC": roc_auc}, step=epoch)

    print(f"ROC AUC: {roc_auc}")
    return average_loss, roc_auc


def _get_dataset(dataset: str, data_dir: str):
    """
    Filter values from MoleculeNet dataset where the data value is empty.
    Relevant for the BBBP dataset.
    """
    return MoleculeNet(root=data_dir, name=dataset, pre_filter=_filter_empty_data)


def _filter_empty_data(data) -> bool:
    return data.x.size()[0] != 0


def _get_dataloaders(dataset, batch_size):
    train_dataset, test_dataset = random_split(dataset=dataset, lengths=[0.8, 0.2])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def _get_num_output_tasks(dataset):
    return torch.numel(dataset[0].y)


def _initialize_finetune_model(pretrain_model, out_dim: int, freeze_pretrain: bool):
    model = FinetuneModel(pretrain_model, out_dim=out_dim)
    if freeze_pretrain:
        model.pretrain_model.freeze()

    return model
