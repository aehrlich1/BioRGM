import warnings
from pathlib import Path

import torch
import wandb
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import roc_auc_score
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torchinfo import summary

from src.model import FinetuneModel, CategoricalEncodingModel
from src.pretrain import Pretrain
from src.utils import save_dict_to_yaml, make_combinations, PerformanceTracker


class FinetuneDispatcher:
    def __init__(self, params: dict, data_dir: str) -> None:
        self.params = params
        self.data_dir = data_dir

    def start(self) -> None:
        with ProcessPoolExecutor(max_workers=8) as executor:
            for pretrain_model_name in self.params["pretrain_models"]:
                finetune_dir: Path = (
                    Path(self.data_dir) / "models" / pretrain_model_name / "finetune"
                )
                finetune_dir.mkdir(exist_ok=True)

                config_finetune: dict = self.params.copy()
                config_finetune.pop("pretrain_models")
                config_finetune["pretrain_model"] = pretrain_model_name
                save_dict_to_yaml(config_finetune, finetune_dir / "config_finetune.yml")

                # create cross_configs (exclude datasets) as a list
                config_combinations = make_combinations(
                    config_finetune, exclude_key="datasets"
                )

                for conf_iteration_num, config_run in enumerate(config_combinations):
                    # create iterated subdir for each conf
                    conf_dir: Path = finetune_dir / f"conf_{conf_iteration_num:02d}"
                    conf_dir.mkdir()

                    # save config file
                    save_dict_to_yaml(config_run, conf_dir / "config_run.yml")

                    for run in range(config_run["runs"]):
                        # create run_dir
                        run_dir: Path = conf_dir / f"run_{run:02d}"
                        run_dir.mkdir()

                        for dataset in self.params["datasets"]:
                            dataset_dir = run_dir / dataset
                            dataset_dir.mkdir()

                            config_single = config_run.copy()
                            config_single.pop("datasets")
                            config_single.pop("runs")
                            config_single["dataset"] = dataset

                            # save config
                            save_dict_to_yaml(
                                config_single, dataset_dir / "config_single.yml"
                            )

                            # create Tracker and inject into Finetune
                            performance_tracker = PerformanceTracker(
                                tracking_dir=dataset_dir
                            )

                            finetune = Finetune(
                                config_single, self.data_dir, performance_tracker
                            )
                            executor.submit(finetune.train)


class Finetune:
    def __init__(
        self,
        params: dict = None,
        data_dir=None,
        performance_tracker: PerformanceTracker = None,
    ) -> None:
        self.params = params
        self.data_dir = data_dir
        self.performance_tracker = performance_tracker
        self.pretrain_model_dir = None
        self.dataset = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None
        self.device = None
        self.finetune_model = None
        self.loss_fn = None
        self.num_output_tasks = None
        self.optimizer = None
        self.pretrain_model = None

        self._initialize()

    def _initialize(self) -> None:
        # self._initialize_wandb()
        self._initialize_device()
        self._initialize_dataset()
        self._initialize_dataloaders()
        self._initialize_num_output_tasks()
        self._initialize_pretrain_model()
        self._initialize_finetune_model()
        self._initialize_optimizer()
        self._initialize_loss_fn()

    def train(self) -> None:
        self.finetune_model.train()

        for epoch in range(self.params["epochs"]):
            print(f"\nEpoch {epoch + 1}\n-------------------------------\n")
            self.performance_tracker.log({"epoch": epoch})
            self._train_loop(epoch)
            self._valid_loop(epoch)
            self._test_loop()
            self.performance_tracker.update_early_loss_state()

            if self.performance_tracker.early_stop: break

        self.performance_tracker.save_performance()
        # wandb.finish()

    def _initialize_wandb(self) -> None:
        wandb.init(project="BioRGM", config=self.params, mode="online", reinit=True)

    def _initialize_device(self) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"From Finetune, using device: {self.device}")

    def _initialize_dataset(self) -> None:
        molecule_net_data_dir = Path(self.data_dir) / "molecule_net"
        self.dataset = self._get_dataset(molecule_net_data_dir=molecule_net_data_dir)

    def _initialize_dataloaders(self) -> None:
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = (
            self._get_dataloaders()
        )

    def _initialize_num_output_tasks(self) -> None:
        self.num_output_tasks = torch.numel(self.dataset[0].y)

    def _initialize_pretrain_model(self) -> None:
        pretrain = Pretrain(data_dir=self.data_dir)

        if self.params["pretrain_model"] is None:
            encoder_model = CategoricalEncodingModel().to(self.device)
            # TODO: Add parameter within config_finetune.yml for pretrain dropout
            pretrain.load_random_model(
                encoder_model=encoder_model, dim_h=self.params["dim_h"], dropout=0.1
            )
        else:
            pretrain.load_pretrained_model(self.params["pretrain_model"])

        self.pretrain_model = pretrain.model

    def _initialize_finetune_model(self) -> None:
        model = FinetuneModel(self.pretrain_model, out_dim=self.num_output_tasks).to(
            self.device
        )
        if self.params["freeze_pretrain"]:
            model.pretrain_model.freeze()

        summary(model)
        self.finetune_model = model
        self.finetune_model.to(self.device)

    def _initialize_optimizer(self) -> None:
        self.optimizer = Adam(self.finetune_model.parameters(), lr=self.params["lr"])

    def _initialize_loss_fn(self) -> None:
        self.loss_fn = BCELoss()

    def _get_dataloaders(self):
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset=self.dataset, lengths=[0.8, 0.1, 0.1]
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.params["batch_size"], shuffle=True
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=self.params["batch_size"], shuffle=False
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.params["batch_size"], shuffle=False
        )

        return train_dataloader, valid_dataloader, test_dataloader

    def _get_dataset(self, molecule_net_data_dir):
        """
        Ignore the warning that incorrect smiles (mol object cannot be formed by RDKit),
        were omitted for the dataset.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            molecule_net_dataset = MoleculeNet(
                root=molecule_net_data_dir,
                name=self.params["dataset"],
            )

        return molecule_net_dataset

    def _train_loop(self, epoch) -> None:
        self.finetune_model.train()
        batch_loss = 0
        y_true, y_pred = [], []

        for data in self.train_dataloader:
            data = data.to(self.device)
            out = self.finetune_model(data)
            loss = self.loss_fn(out, data.y)
            batch_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            y_true.append(data.y)
            y_pred.append(out.detach())

        y_true = torch.cat(y_true).cpu().numpy()
        y_pred = torch.cat(y_pred).cpu().numpy()

        average_loss = batch_loss / len(self.train_dataloader)
        roc_auc = roc_auc_score(y_true, y_pred)
        # wandb.log({"train loss": average_loss}, step=epoch)
        # wandb.log({"train ROC AUC": roc_auc}, step=epoch)

        self.performance_tracker.log({"train_loss": average_loss})
        self.performance_tracker.log({"train_roc_auc": roc_auc})

    def _valid_loop(self, epoch) -> None:
        self.finetune_model.eval()
        batch_loss = 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for data in self.valid_dataloader:
                data = data.to(self.device)
                out = self.finetune_model(data)
                loss = self.loss_fn(out, data.y)
                batch_loss += loss.item()
                y_true.append(data.y)
                y_pred.append(out)

        y_true = torch.cat(y_true).cpu().numpy()
        y_pred = torch.cat(y_pred).cpu().numpy()

        average_loss = batch_loss / len(self.valid_dataloader)
        roc_auc = roc_auc_score(y_true, y_pred)

        # wandb.log({"valid loss": average_loss}, step=epoch)
        # wandb.log({"valid ROC AUC": roc_auc}, step=epoch)

        self.performance_tracker.log({"valid_loss": average_loss})
        self.performance_tracker.log({"valid_roc_auc": roc_auc})

        print(f"ROC AUC Valid: {roc_auc}")

    def _test_loop(self) -> None:
        self.finetune_model.eval()
        batch_loss = 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for data in self.test_dataloader:
                data = data.to(self.device)
                out = self.finetune_model(data)
                loss = self.loss_fn(out, data.y)
                batch_loss += loss.item()
                y_true.append(data.y)
                y_pred.append(out)

        y_true = torch.cat(y_true).cpu().numpy()
        y_pred = torch.cat(y_pred).cpu().numpy()

        average_loss = batch_loss / len(self.test_dataloader)
        roc_auc = roc_auc_score(y_true, y_pred)

        # wandb.log({"test loss": average_loss}, step=epoch)
        # wandb.log({"test ROC AUC": roc_auc}, step=epoch)

        self.performance_tracker.log({"test_loss": average_loss})
        self.performance_tracker.log({"test_roc_auc": roc_auc})

        print(f"ROC AUC Test: {roc_auc}")
