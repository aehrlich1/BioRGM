from src.models.pretrain import Pretrain
from src.models.finetune import finetune


def test_finetune():
    pretrain = Pretrain()
    pretrain.load_random_model(dim_h=128, dropout=0.1)

    finetune(pretrain.model, "BACE", "../data/molecule_net")
