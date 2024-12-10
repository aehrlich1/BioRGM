from src.models.model import CategoricalEncodingModel, OneHotEncoderModel, EcfpModel
from src.models.pretrain import Pretrain
from src.models.finetune import finetune, FinetuneParams


def test_finetune():
    encoder_model = CategoricalEncodingModel()
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
        dataset="BACE",
        epochs=200,
        freeze_pretrain=False,
        lr=1e-5,
    )

    finetune(pretrain_model, finetune_params, "../data/molecule_net")
