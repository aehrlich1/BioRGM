import argparse
from pathlib import Path

import torch.multiprocessing as mp

from src.augmentation import Augmentation
from src.finetune import FinetuneDispatcher
from src.pretrain import PretrainDispatcher
from src.utils import load_yaml_to_dict


def main(args: dict) -> None:
    config_filename = args["config_filename"]
    data_dir = args["data_dir"]

    params: dict = load_yaml_to_dict(config_filename)
    print(f"Data directory is: {data_dir}")

    match params["task"]:
        case "augmentation":
            augmentation = Augmentation(params, data_dir)
            augmentation.start()
        case "pretrain":
            pretrain_dispatcher = PretrainDispatcher(params, data_dir)
            pretrain_dispatcher.start()
        case "finetune":
            finetune_dispatcher = FinetuneDispatcher(params, data_dir)
            finetune_dispatcher.start()
            finetune_dispatcher.data_evaluation()

    print("Done")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Config file and data directory.")
    parser.add_argument(
        "--config_filename",
        type=str,
        required=True,
        help="File name (including extension) of the yaml configuration file.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=False,
        default=str(Path(__file__).parent / "data"),
        help="Absolute path of the directory of the dataset.",
    )

    input_args = parser.parse_args()
    input_args_dict = vars(input_args)

    main(input_args_dict)
