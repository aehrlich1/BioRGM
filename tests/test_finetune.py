from src.models.pretrain import Pretrain
from src.models.finetune import finetune, FinetuneParams


def test_finetune():
    pretrain = Pretrain()
    pretrain.load_random_model(dim_h=128, dropout=0.1)
    pretrain_model = pretrain.model

    finetune_params = FinetuneParams(
        batch_size=32,
        dataset="BACE",
        encoding="embedding",
        epochs=200,
        freeze_pretrain=False,
        lr=1e-5,
    )

    finetune(pretrain_model, finetune_params, "../data/molecule_net")
