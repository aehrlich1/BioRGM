import os
import sys
import logging.config
from src.augmentations.augmentation import Augmentation
from config.reactions import reaction_smarts_list
from config.pretrain import params
from src.models.pretrain import Pretrain


def main(DATA_DIR):
    DATA_DIR = os.path.abspath(DATA_DIR)
    print(f"Data directory is: {DATA_DIR}")

    # Augmentation
    augmentation = Augmentation(
        input_filename="pubchem_1k.txt", root_data_dir=DATA_DIR, processes=None
    )
    augmentation.register_reaction_smarts(reaction_smarts_list)
    augmentation.start()

    # Pretraining
    pretrain = Pretrain(params, DATA_DIR)
    pretrain.initialize_for_training()
    pretrain.train()

    print("Done")


if __name__ == "__main__":
    DATA_DIR = sys.argv[1]
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    logging.config.fileConfig("./config/logging.conf")

    main(DATA_DIR)
