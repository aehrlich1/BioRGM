import os
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


class Checkpoint:
    def __init__(self, data_dir, params):
        self.data_dir = data_dir
        self.params = params
        self.output_dir = None

        self._initialize_directory()

    def _initialize_directory(self):
        os.makedirs(os.path.join(self.data_dir, "models"), exist_ok=True)
        self.output_dir = tempfile.mkdtemp(dir=os.path.join(self.data_dir, "models"))
        print(f"Created directory {self.output_dir}")
        output = {**self.params}

        with open(self.output_dir + "/config_pretrain.yml", "w") as outfile:
            yaml.dump(output, outfile, default_flow_style=False)

    def save(self, model, epoch):
        model_path = os.path.join(self.output_dir, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"\nSaved model to {model_path}")
