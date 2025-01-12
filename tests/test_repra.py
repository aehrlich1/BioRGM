import os
from src.repra import Repra
from src.model import ExtendedConnectivityFingerprintModel
from src.pretrain import Pretrain


def test_repra_fp():
    model = ExtendedConnectivityFingerprintModel()
    repra = Repra(model, "ESOL", "../data/molecule_net")
    improvement_rate, average_deviation = repra.analyze()
    print(
        f"\nimprovement rate: {improvement_rate}, average deviation: {average_deviation}"
    )
    assert improvement_rate == 2.0


def test_repra_random_model():
    pretrain = Pretrain()
    pretrain.load_random_model(dim_h=128, dropout=0.1)
    pretrain.model.eval()

    repra = Repra(pretrain.model, "ESOL", "../data/molecule_net")
    improvement_rate, average_deviation = repra.analyze()
    print(
        f"\nimprovement rate: {improvement_rate}, average deviation: {average_deviation}"
    )
    assert repra


def test_repra_pretrained():
    data_dir = os.environ.get("DATA_DIR", "../data")
    pretrain = Pretrain(data_dir=data_dir)
    pretrain.load_pretrained_model("floral-wind-18")
    pretrain.model.eval()

    repra = Repra(pretrain.model, "ESOL", "../data/molecule_net")
    improvement_rate, average_deviation = repra.analyze()
    print(
        f"\nimprovement rate: {improvement_rate}, average deviation: {average_deviation}"
    )
    assert repra
