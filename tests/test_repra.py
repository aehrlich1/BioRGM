from src.analysis.repra import Repra
from src.models.model import ExtendedConnectivityFingerprintModel
from src.models.pretrain import Pretrain


def test_repra():
    pretrain = Pretrain(data_dir="../data")
    pretrain.load_pretrained_model("dauntless-planet-7")
    pretrain.model.eval()

    # 2. perform repra analysis
    repra = Repra(pretrain.model, ["ESOL"], "../data/molecule_net")
    assert True

def test_repra_ecfp():
    model = ExtendedConnectivityFingerprintModel()
    repra = Repra(model, ["ESOL"], "../data/molecule_net")
    assert True