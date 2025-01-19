import os
from pathlib import Path

import torch
import yaml
import tempfile

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
