import torch
from src.model import CategoricalEncodingModel, OneHotEncoderModel, EcfpModel
from src.pretrain import Pretrain
from src.finetune import finetune, FinetuneParams


def test_finetune_model_with_random_pretrain_model_and_categorical_encoding():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    encoder_model = CategoricalEncodingModel().to(device)
    pretrain = Pretrain()

    pretrain.load_random_model(encoder_model=encoder_model, dim_h=128, dropout=0.1)
    pretrain_model = pretrain.model.to(device)

    finetune_params = FinetuneParams(
        batch_size=32,
        dataset="BACE",
        epochs=300,
        freeze_pretrain=True,
        lr=1e-5,
    )

    finetune(pretrain_model, finetune_params, "../data/molecule_net")


def test_finetune_pretrain():
    # pretrain = Pretrain(data_dir="../data")
    pretrain = Pretrain(data_dir="/mnt/data/anatole93dm/BioRGM/data")

    pretrain.load_pretrained_model("eager-firebrand-39")
    pretrain_model = pretrain.model

    finetune_params = FinetuneParams(
        batch_size=32,
        dataset="HIV",
        epochs=200,
        freeze_pretrain=True,
        lr=1e-5,
    )

    finetune(pretrain_model, finetune_params, "../data/molecule_net")

def test_finetune_one_hot():
    encoder_model = OneHotEncoderModel()
    pretrain = Pretrain()

    pretrain.load_random_model(encoder_model=encoder_model, dim_h=128, dropout=0.1)
    pretrain_model = pretrain.model

    finetune_params = FinetuneParams(
        batch_size=32,
        dataset="BACE",
        epochs=200,
        freeze_pretrain=True,
        lr=1e-5,
    )

    finetune(pretrain_model, finetune_params, "../data/molecule_net")


def test_finetune_ecfp():
    pretrain_model = EcfpModel()

    finetune_params = FinetuneParams(
        batch_size=32,
        dataset="ClinTox",
        epochs=5000,
        freeze_pretrain=False,
        lr=1e-5,
    )

    finetune(pretrain_model, finetune_params, "../data/molecule_net")
