import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torchinfo import summary

import wandb
from biorgm.model import CategoricalEncodingModel, FinetuneModel
from biorgm.pretrain import Pretrain
from biorgm.utils import (
    PerformanceTracker,
    generate_random_alphanumeric,
    make_combinations,
    save_dict_to_csv,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dim_h", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="BACE")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--freeze_pretrain", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--pretrain_model", type=str, default=None)
    parser.add_argument("--runs", type=int, default=3)

    args = parser.parse_args()

    data_dir = Path("data")
    params: dict = vars(args)

    tracking_dir = Path("checkpoints/finetuned") / generate_random_alphanumeric(8)
    tracking_dir.mkdir(exist_ok=True)
    id_run = "1"

    pt = PerformanceTracker(tracking_dir=tracking_dir, id_run=id_run)
    finetune = Finetune(params, pt, data_dir, None)
    finetune.train()


class Finetune:
    def __init__(
        self,
        params: dict,
        performance_tracker: PerformanceTracker,
        data_dir=None,
        queue=None,
    ) -> None:
        self.params: dict = params
        self.data_dir = data_dir
        self.performance_tracker = performance_tracker
        self.queue = queue
        self.pretrain_model_dir = None
        self.dataset = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None
        self.device = None
        self.finetune_model = None
        self.loss_fn = None
        self.num_output_tasks = None
        self.optimizer = None
        self.pretrain_model = None

        self._initialize()

    def _initialize(self) -> None:
        # self._initialize_wandb()
        self._initialize_device()
        self._initialize_dataset()
        self._initialize_dataloaders()
        self._initialize_num_output_tasks()
        self._initialize_pretrain_model()
        self._initialize_finetune_model()
        self._initialize_optimizer()
        self._initialize_loss_fn()

    def train(self) -> None:
        self.finetune_model.train()

        for epoch in range(self.params["epochs"]):
            print(f"\nEpoch {epoch + 1}\n-------------------------------\n")
            self.performance_tracker.log({"epoch": epoch})
            self._train_loop(epoch)
            self._valid_loop(epoch)
            self._test_loop()
            self.performance_tracker.update_early_loss_state()

            if self.performance_tracker.early_stop:
                break

        self.performance_tracker.save_performance()
        final_results = self.performance_tracker.get_results()
        submit = self.params | final_results
        save_dict_to_csv([submit], self.performance_tracker.tracking_dir / "results.csv")
        # self.queue.put(submit)

        # save results

        # wandb.finish()

    def _initialize_wandb(self) -> None:
        wandb.init(project="BioRGM", config=self.params, mode="online", reinit=True)

    def _initialize_device(self) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"From Finetune, using device: {self.device}")

    def _initialize_dataset(self) -> None:
        molecule_net_data_dir = Path(self.data_dir) / "molecule_net"
        self.dataset = self._get_dataset(molecule_net_data_dir=molecule_net_data_dir)

    def _initialize_dataloaders(self) -> None:
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = self._get_dataloaders()

    def _initialize_num_output_tasks(self) -> None:
        self.num_output_tasks = torch.numel(self.dataset[0].y)

    def _initialize_pretrain_model(self) -> None:
        pretrain = Pretrain(None)

        if self.params["pretrain_model"] is None:
            encoder_model = CategoricalEncodingModel().to(self.device)
            # TODO: Add parameter within config_finetune.yml for pretrain dropout
            pretrain.load_random_model(
                encoder_model=encoder_model, dim_h=self.params["dim_h"], dropout=0.1
            )
        else:
            pretrain.load_pretrained_model(self.params["pretrain_model"])
            self.params["dim_h"] = pretrain.get_dim_h()

        self.pretrain_model = pretrain.model

    def _initialize_finetune_model(self) -> None:
        model = FinetuneModel(
            self.pretrain_model,
            in_dim=self.params["dim_h"],
            out_dim=self.num_output_tasks,
        ).to(self.device)
        if self.params["freeze_pretrain"]:
            model.pretrain_model.freeze()

        summary(model)
        self.finetune_model = model
        self.finetune_model.to(self.device)

    def _initialize_optimizer(self) -> None:
        self.optimizer = Adam(self.finetune_model.parameters(), lr=self.params["lr"])

    def _initialize_loss_fn(self) -> None:
        self.loss_fn = BCELoss()

    def _get_dataloaders(self):
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset=self.dataset, lengths=[0.8, 0.1, 0.1]
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.params["batch_size"], shuffle=True
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=self.params["batch_size"], shuffle=False
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.params["batch_size"], shuffle=False
        )

        return train_dataloader, valid_dataloader, test_dataloader

    def _get_dataset(self, molecule_net_data_dir):
        """
        Ignore the warning that incorrect smiles (mol object cannot be formed by RDKit),
        were omitted for the dataset.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            molecule_net_dataset = MoleculeNet(
                root=molecule_net_data_dir,
                name=self.params["dataset"],
            )

        return molecule_net_dataset

    def _train_loop(self, epoch) -> None:
        self.finetune_model.train()
        batch_loss = 0
        y_true, y_pred = [], []

        for data in self.train_dataloader:
            data = data.to(self.device)
            out = self.finetune_model(data)
            loss = self.loss_fn(out, data.y)
            batch_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            y_true.append(data.y)
            y_pred.append(out.detach())

        y_true = torch.cat(y_true).cpu().numpy()
        y_pred = torch.cat(y_pred).cpu().numpy()

        average_loss = batch_loss / len(self.train_dataloader)
        roc_auc = roc_auc_score(y_true, y_pred)
        # wandb.log({"train loss": average_loss}, step=epoch)
        # wandb.log({"train ROC AUC": roc_auc}, step=epoch)

        self.performance_tracker.log({"train_loss": average_loss})
        self.performance_tracker.log({"train_roc_auc": roc_auc})

    def _valid_loop(self, epoch) -> None:
        self.finetune_model.eval()
        batch_loss = 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for data in self.valid_dataloader:
                data = data.to(self.device)
                out = self.finetune_model(data)
                loss = self.loss_fn(out, data.y)
                batch_loss += loss.item()
                y_true.append(data.y)
                y_pred.append(out)

        y_true = torch.cat(y_true).cpu().numpy()
        y_pred = torch.cat(y_pred).cpu().numpy()

        average_loss = batch_loss / len(self.valid_dataloader)
        roc_auc = roc_auc_score(y_true, y_pred)

        # wandb.log({"valid loss": average_loss}, step=epoch)
        # wandb.log({"valid ROC AUC": roc_auc}, step=epoch)

        self.performance_tracker.log({"valid_loss": average_loss})
        self.performance_tracker.log({"valid_roc_auc": roc_auc})

        print(f"ROC AUC Valid: {roc_auc}")

    def _test_loop(self) -> None:
        self.finetune_model.eval()
        batch_loss = 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for data in self.test_dataloader:
                data = data.to(self.device)
                out = self.finetune_model(data)
                loss = self.loss_fn(out, data.y)
                batch_loss += loss.item()
                y_true.append(data.y)
                y_pred.append(out)

        y_true = torch.cat(y_true).cpu().numpy()
        y_pred = torch.cat(y_pred).cpu().numpy()

        average_loss = batch_loss / len(self.test_dataloader)
        roc_auc = roc_auc_score(y_true, y_pred)

        # wandb.log({"test loss": average_loss}, step=epoch)
        # wandb.log({"test ROC AUC": roc_auc}, step=epoch)

        self.performance_tracker.log({"test_loss": average_loss})
        self.performance_tracker.log({"test_roc_auc": roc_auc})

        print(f"ROC AUC Test: {roc_auc}")


if __name__ == "__main__":
    main()
