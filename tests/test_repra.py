from src.analysis.repra import Repra
from src.analysis.repra_v2 import RepraV2
from src.models.model import ExtendedConnectivityFingerprintModel
from src.models.pretrain import Pretrain


def test_repra():
    pretrain = Pretrain(data_dir="../data")
    pretrain.load_pretrained_model("dauntless-planet-7")
    pretrain.model.eval()

    repra = Repra(pretrain.model, ["ESOL"], "../data/molecule_net")
    assert repra


def test_repra_ecfp():
    model = ExtendedConnectivityFingerprintModel()
    repra = Repra(model, ["ESOL"], "../data/molecule_net")
    assert repra


def test_repra_random_model():
    pretrain = Pretrain()
    pretrain.load_random_model(dim_h=128, dropout=0.1)
    pretrain.model.eval()

    repra = Repra(pretrain.model, ["ESOL"], "../data/molecule_net")
    assert repra

def test_repra_v2():
    pretrain = Pretrain(data_dir="../data")
    pretrain.load_pretrained_model("dauntless-planet-7")
    pretrain.model.eval()

    repra_v2 = RepraV2(pretrain.model, "ESOL", "../data/molecule_net")
    improvement_rate, average_deviation = repra_v2.analyze()
    print(f"improvement rate: {improvement_rate}, average deviation: {average_deviation}")
    assert repra_v2

def test_repra_v2_ecfp():
    model = ExtendedConnectivityFingerprintModel()
    repra_v2 = RepraV2(model, "ESOL", "../data/molecule_net")
    improvement_rate, average_deviation = repra_v2.analyze()
    assert improvement_rate == 2.0


def test_repra_v2_random():
    pretrain = Pretrain()
    pretrain.load_random_model(dim_h=128, dropout=0.1)
    pretrain.model.eval()

    repra_v2 = RepraV2(pretrain.model, "ESOL", "../data/molecule_net")
    improvement_rate, average_deviation = repra_v2.analyze()
    print(f"improvement rate: {improvement_rate}, average deviation: {average_deviation}")
    assert repra_v2