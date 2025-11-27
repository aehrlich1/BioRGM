import os
import random
from multiprocessing import Pool
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Mol

from config.reactions import reaction_smarts_list
from src.utils import neutralize_smiles

RDLogger.DisableLog("rdApp.*")


class Augmentation:
    def __init__(self, params: dict, data_dir: str):
        self.input_filename = Path(params["input_filename"]).stem
        self.input_file = Path(data_dir) / "raw" / params["input_filename"]
        self.data_dir = data_dir
        self.processes = params["processes"]
        self.reaction_smarts = reaction_smarts_list

    def start(self) -> None:
        """
        process the file provided to the constructor as filename.
        Multiprocessing is used to drastically improve the runtime.
        """
        with open(self.input_file, "r") as file:
            with Pool(processes=self.processes) as pool:
                products: list[list[str]] = pool.imap(self.process_smiles, iterable=enumerate(file))
                products = [row for row in products if row is not None]
                self.save_to_file(
                    f"{self.input_filename}_processed.csv",
                    self.generate_header(),
                    products,
                )

                triplets: list[list[list[str]]] = pool.map(
                    self.generate_triplets, iterable=products
                )
                flat_list: list[list[str]] = []
                for row in triplets:
                    flat_list.extend(row)
                self.save_to_file(
                    f"{self.input_filename}_triplets.csv", "label,smiles\n", flat_list
                )

    def register_reaction_smarts(self, rxn_smarts: list[str]):
        self.reaction_smarts.extend(rxn_smarts)

    def process_smiles(self, idx_smiles: tuple[int, str]) -> list[str] | None:
        """
        Takes a single smiles string as input, neutralizes it, and performs all reactions
        defined in reaction_smarts on it.
        """
        idx, smiles = idx_smiles
        try:
            smiles = smiles.strip()  # remove \n character
            smiles = Chem.CanonSmiles(smiles)
            smiles_neutral = neutralize_smiles(smiles)

            products: list = [idx, smiles, smiles_neutral]

            # returns None if there is an error
            reactant: Mol | None = Chem.MolFromSmiles(smiles_neutral)
            if reactant is None:
                return None

            for rxn_smarts in self.reaction_smarts:
                product: str = self.run_reaction(smiles_neutral, reactant, rxn_smarts)
                products.append(product)

        except Exception:
            print(
                f'Error while processing {smiles} in "process_smiles" function. '
                f"Returning None instead."
            )
            return None

        return products

    def generate_header(self) -> str:
        reaction_ids = ",".join(f"rxn_{i:03}" for i in range(len(self.reaction_smarts)))
        header = f"id,smiles,smiles_neutral,{reaction_ids}\n"
        return header

    def save_to_file(self, filename, header: str, data: list[list[str]]) -> None:
        folder = Path(self.data_dir) / "processed"
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, filename)

        with open(filename, "w") as f:
            f.write(header)
            for line_list in data:
                f.write(",".join(map(str, line_list)))
                f.write("\n")

        print(f"Saved {filename}")

    @staticmethod
    def generate_triplets(products: list[str]) -> list[list[str]]:
        label = products[0]
        return [[label, smiles] for smiles in products[2:] if smiles != ""]

    @staticmethod
    def run_reaction(smiles: str, reactant: Mol, reaction_smarts: str) -> str:
        try:
            reaction = AllChem.ReactionFromSmarts(reaction_smarts)
            products = reaction.RunReactants((reactant,))
            if len(products) == 0:
                return ""
            product: Mol = products[random.randint(0, len(products) - 1)][0]

            Chem.SanitizeMol(product)
            product_smiles: str = Chem.MolToSmiles(product)
        except Chem.KekulizeException as e:
            print(
                f"{e}. Could not kekulize: Returned empty str. Original: {smiles}. "
                f"Caused by {reaction_smarts}"
            )
            return ""
        except ValueError as e:
            print(
                f"{e}. Value Error exception for: {smiles} returned empty str. "
                f"Caused by {reaction_smarts}"
            )
            return ""

        return product_smiles if product_smiles != smiles else ""
