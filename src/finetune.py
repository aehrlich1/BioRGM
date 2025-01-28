import warnings
from pathlib import Path

import torch
import wandb
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torchinfo import summary

from src.model import FinetuneModel, CategoricalEncodingModel
from src.pretrain import Pretrain
from src.utils import (
    save_dict_to_yaml,
    make_combinations,
    PerformanceTracker,
    generate_random_alphanumeric,
    save_dict_to_csv,
)


class FinetuneDispatcher:
    def __init__(self, params: dict, data_dir: str) -> None:
        self.params = params
        self.data_dir = data_dir

    def start(self) -> None:
        with Manager() as manager:
            queue = manager.Queue()
            with ProcessPoolExecutor(max_workers=None) as executor:
                pretrain_model_names: list[str] = self._get_pretrain_model_names()
                self.params.update({"pretrain_models": pretrain_model_names})
                self.params.pop("task")

                for pretrain_model_name in pretrain_model_names:
                    finetune_dir: Path = self._create_finetune_dir(pretrain_model_name)
                    figs_dir: Path = finetune_dir / "figs"
                    figs_dir.mkdir(exist_ok=True)
                    # save_dict_to_yaml(config_combinations, finetune_dir)

                    config_combinations: list[dict] = self._create_config_combinations(
                        pretrain_model_name
                    )
                    for config_combination in config_combinations:
                        for run in range(config_combination["runs"]):
                            id_run = generate_random_alphanumeric(8)
                            config_combination.update({"id_run": id_run})
                            performance_tracker = PerformanceTracker(figs_dir, id_run)
                            finetune = Finetune(
                                config_combination,
                                self.data_dir,
                                performance_tracker,
                                queue,
                            )
                            executor.submit(finetune.train)

            executor.shutdown()
            result = []
            while not queue.empty():
                result.append(queue.get())

            finetune_results_path = (
                Path(self.data_dir) / "models" / "finetune_overview.csv"
            )

            save_dict_to_csv(result, finetune_results_path)
            print(f"Finetune results saved to {finetune_results_path}")

    def _get_pretrain_model_names(self) -> list:
        if self.params["pretrain_models"] is None:
            models_dir: Path = Path(self.data_dir) / "models"
            pretrain_model_names = [
                pretrain_model.stem
                for pretrain_model in models_dir.iterdir()
                if pretrain_model.is_dir()
            ]
            return pretrain_model_names

        return self.params["pretrain_models"]

    def data_evaluation(self) -> None:
        """
        Iterate through
        """
        column_names = [
            "dataset",
            "conf",
            "train_loss",
            "train_roc_auc",
            "valid_loss",
            "valid_roc_auc",
            "test_loss",
            "test_roc_auc",
        ]

        models_dir: Path = Path("./data/models")
        model_dfs_list = []
        for pretrain_model_name in self._get_pretrain_model_names():
            model_dir: Path = models_dir / pretrain_model_name
            finetune_dir: Path = model_dir / "finetune"
            finetune_df = pd.DataFrame()

            # get all conf directories
            conf_dirs: list[Path] = [
                d
                for d in finetune_dir.iterdir()
                if d.name.startswith("conf_") and d.is_dir()
            ]
            for conf_dir in conf_dirs:
                conf_list = []
                run_dirs: list[Path] = [
                    d
                    for d in conf_dir.iterdir()
                    if d.name.startswith("run_") and d.is_dir()
                ]
                for run_dir in run_dirs:
                    run_list = []
                    dataset_dirs: list[Path] = [
                        d for d in run_dir.iterdir() if d.is_dir()
                    ]
                    for dataset_dir in dataset_dirs:
                        perf_list: list = self.read_last_row_csv(
                            dataset_dir / "performance.csv"
                        )
                        run_list.append([dataset_dir.name, conf_dir.name] + perf_list)
                    # save run_xx_results.csv
                    # conf_list.extend(run_list)
                    run_df = pd.DataFrame(run_list, columns=column_names)
                    run_df.to_csv(run_dir / "run_results.csv", index=False)
                    conf_list.extend(run_list)
                # conf_xx_results.csv
                conf_df = pd.DataFrame(conf_list, columns=column_names)
                conf_df = conf_df.groupby(["dataset", "conf"]).agg(["mean", "std"])
                conf_df = conf_df.reset_index()
                # Save to CSV with flattened columns
                conf_df.columns = pd.MultiIndex.from_tuples(conf_df.columns)
                conf_df.columns = [
                    "_".join(col).strip() if i > 1 else col[0]
                    for i, col in enumerate(conf_df.columns)
                ]
                conf_df.to_csv(conf_dir / "conf_results.csv", index=False)
                finetune_df = pd.concat([finetune_df, conf_df])
            # finetune_results.csv
            finetune_df.sort_values(by=["dataset", "conf"], inplace=True)
            finetune_df.to_csv(finetune_dir / "finetune_results.csv", index=False)
            finetune_df = finetune_df[
                ["dataset", "conf", "test_roc_auc_mean", "test_roc_auc_std"]
            ]
            finetune_df["model"] = pretrain_model_name
            model_dfs_list.append(finetune_df)

        # model_results.csv
        model_df = pd.concat(model_dfs_list, ignore_index=True)
        # Pivot the DataFrame
        model_df = model_df.pivot_table(index=["model", "conf"], columns="dataset")

        # Flatten the MultiIndex columns
        model_df.columns = [f"{col[1]}_{col[0]}" for col in model_df.columns]

        # Reset the index to get the desired format
        model_df = model_df.reset_index()

        # Rename columns to match the desired output
        model_df.columns.name = None

        # Sort all columns alphabetically except 'model' and 'conf'
        other_cols = sorted(
            [col for col in model_df.columns if col not in ["model", "conf"]]
        )
        model_df = model_df[["model", "conf"] + other_cols]

        # Add the 'sum_roc_auc' column by summing rows where the column name ends with 'mean'
        mean_columns = [col for col in model_df.columns if col.endswith("mean")]
        model_df["sum_roc_auc"] = model_df[mean_columns].sum(axis=1)

        model_df.to_csv(models_dir / "model_results_overview.csv", index=False)

        # Extract the row with the highest 'sum_roc_auc' for each model
        best_models_df = model_df.loc[model_df.groupby("model")["sum_roc_auc"].idxmax()]
        best_models_df.to_csv(models_dir / "model_results.csv", index=False)

    def data_evaluation_improved(self) -> None:
        # Read the results file
        results_file = Path(self.data_dir) / "models" / "finetune_overview.csv"
        df = pd.read_csv(results_file)
        df.drop(
            columns=[
                "train_loss",
                "train_roc_auc",
                "valid_loss",
                "valid_roc_auc",
                "test_loss",
            ],
            inplace=True,
        )

        # Get all column names except the last one
        columns_to_groupby = df.columns[:-1]

        # Group by all columns except the last one and aggregate the last column
        result = df.groupby(list(columns_to_groupby)).agg(["mean", "std"])
        # print(result)


    @staticmethod
    def read_last_row_csv(file_path: Path) -> list:
        """
        Read last row csv file and remove first column (epoch)
        """
        df = pd.read_csv(file_path)
        return list(df.iloc[-1][1:])

    @staticmethod
    def _create_config_single(config_run, dataset):
        config_single = config_run.copy()
        config_single.pop("datasets")
        config_single.pop("runs")
        config_single["dataset"] = dataset
        return config_single

    def _create_finetune_dir(self, pretrain_model_name) -> Path:
        finetune_dir: Path = (
            Path(self.data_dir) / "models" / pretrain_model_name / "finetune"
        )
        finetune_dir.mkdir(exist_ok=True)
        return finetune_dir

    def _create_config_combinations(self, pretrain_model_name: str) -> list[dict]:
        config_finetune: dict = self.params.copy()
        config_finetune.pop("pretrain_models")
        config_finetune["pretrain_model"] = pretrain_model_name

        # TODO: is there no python function for this?
        return make_combinations(config_finetune)


class Finetune:
    def __init__(
        self,
        params: dict = None,
        data_dir=None,
        performance_tracker: PerformanceTracker = None,
        queue=None,
    ) -> None:
        self.params = params
        self.data_dir = data_dir
        self.performance_tracker = performance_tracker
        self.queue = queue
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

            if self.performance_tracker.early_stop:
                break

        self.performance_tracker.save_performance()
        final_results = self.performance_tracker.get_results()
        submit = self.params | final_results
        self.queue.put(submit)
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

        self.params["dim_h"] = pretrain.get_dim_h()
        self.pretrain_model = pretrain.model

    def _initialize_finetune_model(self) -> None:
        model = FinetuneModel(
            self.pretrain_model,
            in_dim=self.params["dim_h"],
            out_dim=self.num_output_tasks,
        ).to(self.device)
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
