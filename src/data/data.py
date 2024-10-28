import os
import torch_geometric
import torch

from pathlib import Path
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.smiles import from_smiles


def override(method):
    """
    Fix for @override decorator missing for python < 3.12
    """
    return method


class PubchemDataset(InMemoryDataset):
    """
    Override raw_dir and processed_dir in Dataset class to avoid having
    to place files into the 'raw_dir' and 'processed_dir' folders within the
    root directory.
    """

    def __init__(self, root: str, file_name: str):
        self.file_name = file_name
        super().__init__(root)
        self.load(self.processed_paths[0])

    @override
    @property
    def raw_dir(self) -> str:
        return self.root

    @override
    @property
    def processed_dir(self) -> str:
        return self.root

    @property
    def raw_file_names(self) -> str:
        return os.path.join(self.raw_dir, self.file_name)

    @property
    def processed_file_names(self) -> str:
        path = Path(self.raw_file_names).with_suffix(".pt")
        return str(path)

    def process(self) -> None:
        with open(self.raw_paths[0], "r") as f:
            lines = f.read().split("\n")[1:-1]

        data_list = []
        for line in lines:
            label, smiles = line.split(",")

            y = torch.tensor(int(label), dtype=torch.int)
            data: torch_geometric.data.Data = from_smiles(smiles)

            data.y = y
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
