from src.analysis.repra import Repra
from src.models.model import ExtendedConnectivityFingerprintModel
from src.models.pretrain import Pretrain


def test_repra_fp():
    model = ExtendedConnectivityFingerprintModel()
    repra = Repra(model, "ESOL", "../data/molecule_net")
    improvement_rate, average_deviation = repra.analyze()
    print(
        f"improvement rate: {improvement_rate}, average deviation: {average_deviation}"
    )
    assert improvement_rate == 2.0


def test_repra_random_model():
    pretrain = Pretrain()
    pretrain.load_random_model(dim_h=128, dropout=0.1)
    pretrain.model.eval()

    repra = Repra(pretrain.model, "ESOL", "../data/molecule_net")
    improvement_rate, average_deviation = repra.analyze()
    print(
        f"improvement rate: {improvement_rate}, average deviation: {average_deviation}"
    )
    assert repra


def test_repra_pretrained():
    pretrain = Pretrain(data_dir="../data")
    pretrain.load_pretrained_model("dauntless-planet-7")
    pretrain.model.eval()

    repra = Repra(pretrain.model, "ESOL", "../data/molecule_net")
    improvement_rate, average_deviation = repra.analyze()
    print(
        f"improvement rate: {improvement_rate}, average deviation: {average_deviation}"
    )
    assert repra
