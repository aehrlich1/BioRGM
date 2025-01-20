import os
from pathlib import Path

import pandas as pd
import torch
import yaml
import tempfile
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger

un = Uncharger()


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
            self.output_dir_name = tempfile.mkdtemp(
                dir=os.path.join(self.data_dir, "models")
            )
        self.output_dir_name = os.path.join(
            self.data_dir, "models", self.output_dir_name
        )
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
    def __init__(self, tracking_dir: Path):
        self.tracking_dir: Path = tracking_dir
        self.epoch = []
        self.train_loss = []
        self.train_roc_auc = []
        self.test_loss = []
        self.test_roc_auc = []

    def save(self) -> None:
        self.save_to_csv()
        self.save_loss_plot()
        self.save_roc_auc_plot()

    def save_loss_plot(self) -> None:
        loss_plot_path = self.tracking_dir / "loss_plot.pdf"
        fig, ax = plt.subplots(figsize=(10, 6), layout="constrained", dpi=300)
        ax.plot(self.epoch, self.train_loss, self.test_loss)
        ax.grid(True)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Test"])

        fig.savefig(loss_plot_path)
        print(f"Saved loss plot to: {loss_plot_path}")

    def save_roc_auc_plot(self) -> None:
        roc_auc_plot_path = self.tracking_dir / "roc_auc_plot.pdf"
        fig, ax = plt.subplots(figsize=(10, 6), layout="constrained", dpi=300)
        ax.plot(self.epoch, self.train_roc_auc, self.test_roc_auc)
        ax.grid(True)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("ROC AUC")
        ax.legend(["Train", "Test"])

        fig.savefig(roc_auc_plot_path)
        print(f"Saved loss plot to: {roc_auc_plot_path}")

    def save_to_csv(self) -> None:
        df = pd.DataFrame(
            {
                "epoch": self.epoch,
                "train_loss": self.train_loss,
                "train_roc_auc": self.train_roc_auc,
                "test_loss": self.test_loss,
                "test_roc_auc": self.test_roc_auc,
            }
        )
        df.to_csv(self.tracking_dir / "performance.csv", index=False)

    def log(self, data: dict[str, int | float]) -> None:
        for key, value in data.items():
            attr = getattr(self, key)
            attr.append(value)
