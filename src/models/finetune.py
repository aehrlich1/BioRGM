import torch
import wandb
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
from torch.nn import BCELoss
from torch.utils.data import random_split
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

from src.models.model import FinetuneModel


def finetune(pretrain_model, dataset: str, data_dir: str) -> None:
    """
    1. load dataset
    2. get dataloader
    3. load pre-trained model
    4. perform fine-tuning on dataset
    """
    wandb.init(
        project="BioRGM_finetune",
    )

    dataset = _get_dataset(dataset, data_dir)
    train_dataloader, test_dataloader = _get_dataloaders(dataset)
    finetune_model = _initialize_finetune_model(pretrain_model)

    _test_loop(finetune_model, test_dataloader, epoch=0)
    for epoch in range(200):
        print(f"\nEpoch {epoch+1}\n-------------------------------\n")
        _train_loop(finetune_model, train_dataloader, epoch)
        _test_loop(finetune_model, test_dataloader, epoch)


def _train_loop(model, dataloader, epoch):
    model.train()
    loss_fn = BCELoss()
    optimizer = Adam(model.parameters(), lr=0.00001)
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


def _test_loop(model, dataloader, epoch):
    model.eval()
    loss_fn = BCELoss()
    batch_loss = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for data in dataloader:
            out = model(data)
            loss = loss_fn(out, data.y)
            batch_loss += loss.item()
            y_true.append(data.y.view(-1))
            y_pred.append(out.view(-1))

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    average_loss = batch_loss / len(dataloader)
    roc_auc = roc_auc_score(y_true, y_pred)

    wandb.log({"test loss": average_loss}, step=epoch)
    wandb.log({"ROC AUC": roc_auc}, step=epoch)

    print(f"ROC AUC: {roc_auc}")
    return average_loss, roc_auc


def _get_dataset(dataset: str, data_dir: str) -> DataLoader:
    return MoleculeNet(root=data_dir, name=dataset)


def _get_dataloaders(dataset):
    train_dataset, test_dataset = random_split(dataset=dataset, lengths=[0.8, 0.2])

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_dataloader, test_dataloader


def _initialize_finetune_model(pretrain_model):
    return FinetuneModel(pretrain_model)
