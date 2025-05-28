import os
from pathlib import Path
from typing import override

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils.smiles import from_smiles


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

    @property
    @override
    def raw_dir(self) -> str:
        return self.root

    @property
    @override
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
        data_list = []
        with open(self.raw_paths[0], "r") as file:
            next(file)
            lines = file.readlines()
            for line in lines:
                label, smiles = line.split(",")
                smiles = smiles.strip()

                y = torch.tensor(int(label), dtype=torch.int)
                data: Data = from_smiles(smiles)

                data.y = y
                data_list.append(data)

        self.save(data_list, self.processed_paths[0])
