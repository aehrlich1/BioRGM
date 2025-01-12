from src.pretrain import Pretrain

def test_pretrain():
    config = {
        "batch_size": 800,
        "num_workers": 0,
        "dim_h": 128,
        "dropout": 0.1,
        "encoder": "embedding",
        "margin": 0.4,
        "epochs": 100,
        "distance_metric": "cosine",
        "learning_rate": 1e-6,
        "weight_decay": 5e-4,
        "num_samples_per_class": 4,
        "type_of_triplets": "all",
        "file_name": "pubchem_1m_triplets.csv"
    }

    DATA_DIR = "/mnt/data/anatole93dm/BioRGM/data"
    # DATA_DIR = "~/src/BioRGM/data"
    pretrain = Pretrain(config, DATA_DIR)
    pretrain.initialize_for_training()
    pretrain.train()
