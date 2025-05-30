import csv
import os
import random
import string
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger

un = Uncharger()


def generate_random_alphanumeric(length=8) -> str:
    characters = string.ascii_lowercase + string.digits
    random_sequence = "".join(random.choice(characters) for _ in range(length))
    return random_sequence


def save_dict_to_csv(data: list[dict], output_path: Path):
    with open(output_path, "w", newline="") as file:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def neutralize_smiles(smiles: str) -> str:
    try:
        mol: Mol = Chem.MolFromSmiles(smiles)
        neutralized_mol = un.uncharge(mol)
    except Chem.AtomValenceException:
        print(f"Could not neutralize smiles: '{smiles}' Returning empty string.")
        return ""

    neutralized_smiles = Chem.MolToSmiles(neutralized_mol)
    return neutralized_smiles


def read_config_file(file_path) -> dict:
    with open(file_path, "r") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    return params


class Checkpoint:
    def __init__(self, data_dir, params, output_dir_name=None):
        self.data_dir = data_dir
        self.params = params
        self.output_dir_name = output_dir_name

        self._initialize_directory()

    def _create_output_dir(self):
        if self.output_dir_name is None:
            self.output_dir_name = tempfile.mkdtemp(dir=os.path.join(self.data_dir, "models"))
        self.output_dir_name = os.path.join(self.data_dir, "models", self.output_dir_name)
        os.makedirs(self.output_dir_name)

    def _initialize_directory(self):
        os.makedirs(os.path.join(self.data_dir, "models"), exist_ok=True)
        self._create_output_dir()
        output = {**self.params}

        with open(self.output_dir_name + "/config_pretrain.yml", "w") as outfile:
            yaml.dump(output, outfile, default_flow_style=False)

        print(f"Created directory {self.output_dir_name}")

    def save(self, model, epoch):
        model_path = os.path.join(self.output_dir_name, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"\nSaved model to {model_path}")


def load_yaml_to_dict(config_filename: str) -> dict:
    path = Path(".") / "config" / config_filename
    with open(path, "r") as file:
        config: dict = yaml.safe_load(file)

    return config


def save_dict_to_yaml(dict_to_save: dict, filename_path: Path) -> None:
    with open(filename_path, "w") as file:
        yaml.dump(dict_to_save, file, default_flow_style=False)


def make_combinations(dictionary: dict, exclude_key: str = None) -> list[dict]:
    # Start with first key-value pair
    result = [{}]

    for key, value in dictionary.items():
        if key == exclude_key:
            # Add this key with its list value to all existing dictionaries
            for r in result:
                r[key] = value
        else:
            # For all other keys, create combinations
            new_result = []
            values = [value] if not isinstance(value, list) else value
            for r in result:
                for v in values:
                    new_dict = r.copy()
                    new_dict[key] = v
                    new_result.append(new_dict)
            result = new_result

    return result


class PerformanceTracker:
    def __init__(self, tracking_dir: Path, id_run: str):
        self.tracking_dir: Path = tracking_dir
        self.id_run = id_run
        self.epoch = []
        self.train_loss = []
        self.train_roc_auc = []
        self.valid_loss = []
        self.valid_roc_auc = []
        self.test_loss = []
        self.test_roc_auc = []

        self.counter = 0
        self.patience = 15
        self.best_valid_loss = float("inf")
        self.early_stop = False

    def save_performance(self) -> None:
        self.save_to_csv()
        self.save_loss_plot()
        self.save_roc_auc_plot()

    def save_loss_plot(self) -> None:
        loss_plot_path = self.tracking_dir / f"{self.id_run}_loss.pdf"
        fig, ax = plt.subplots(figsize=(10, 6), layout="constrained", dpi=300)
        ax.plot(self.epoch, self.train_loss, self.valid_loss)
        ax.grid(True)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Valid"])

        fig.savefig(loss_plot_path)
        print(f"Saved loss plot to: {loss_plot_path}")

    def save_roc_auc_plot(self) -> None:
        roc_auc_plot_path = self.tracking_dir / f"{self.id_run}_roc_auc.pdf"
        fig, ax = plt.subplots(figsize=(10, 6), layout="constrained", dpi=300)
        ax.plot(self.epoch, self.train_roc_auc, self.valid_roc_auc)
        ax.grid(True)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("ROC AUC")
        ax.legend(["Train", "Valid"])

        fig.savefig(roc_auc_plot_path)
        print(f"Saved ROC AUC plot to: {roc_auc_plot_path}")

    def save_to_csv(self) -> None:
        df = pd.DataFrame(
            {
                "epoch": self.epoch,
                "train_loss": self.train_loss,
                "train_roc_auc": self.train_roc_auc,
                "valid_loss": self.valid_loss,
                "valid_roc_auc": self.valid_roc_auc,
                "test_loss": self.test_loss,
                "test_roc_auc": self.test_roc_auc,
            }
        )
        df.to_csv(self.tracking_dir / f"{self.id_run}.csv", index=False)

    def get_results(self) -> dict[str, float]:
        return {
            "train_loss": self.train_loss[-1],
            "train_roc_auc": self.train_roc_auc[-1],
            "valid_loss": self.valid_loss[-1],
            "valid_roc_auc": self.valid_roc_auc[-1],
            "test_loss": self.test_loss[-1],
            "test_roc_auc": self.test_roc_auc[-1],
        }

    def update_early_loss_state(self) -> None:
        if self.valid_loss[-1] < self.best_valid_loss:
            self.best_valid_loss = self.valid_loss[-1]
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            print("Early stopping triggered.")

    def log(self, data: dict[str, int | float]) -> None:
        for key, value in data.items():
            attr = getattr(self, key)
            attr.append(value)
