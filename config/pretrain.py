params = {
    "batch_size": 800,
    "num_workers": 10,
    "dim_h": 128,
    "dropout": 0.1,
    "encoder": "embedding",
    "margin": 0.4,
    "epochs": 3,
    "distance_metric": "cosine",
    "learning_rate": 1e-5,
    "weight_decay": 5e-4,
    "num_samples_per_class": 4,
    "type_of_triplets": "all",
    "file_name": "pubchem_100k_triplets.csv"
}
