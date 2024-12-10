import sys
import torch
import torch.nn.functional as F
import torch_geometric.utils.smiles as pyg_smiles
from torch import nn
from torch_geometric.nn import GINEConv, global_add_pool
from rdkit import Chem
from rdkit.Chem import AllChem


def mlp(dim_in, dim_out):
    return nn.Sequential(
        nn.Linear(dim_in, dim_out),
        nn.BatchNorm1d(dim_out),
        nn.ReLU(),
        nn.Linear(dim_out, dim_out),
        nn.ReLU(),
    )


class PretrainModel(nn.Module):
    """
    Combines the CategoricalEmbeddingModel and the GIN Model
    """

    def __init__(self, encoder, dim_h, dropout):
        super().__init__()
        dim_in = encoder.get_feature_embedding_dim()
        edge_dim = encoder.get_edge_embedding_dim()

        self.embedding_model = encoder
        self.gin_model = GIN(
            dim_in=dim_in,
            dim_h=dim_h,
            edge_dim=edge_dim,
            dropout=dropout,
        )

    def forward(self, data):
        data = self.embedding_model(data)
        h = self.gin_model(data)
        return h

    def freeze(self):
        for param in self.embedding_model.parameters():
            param.requires_grad = False

        for param in self.gin_model.parameters():
            param.requires_grad = False


class OneHotEncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_node_categories = self._get_num_node_categories()
        self.num_edge_categories = self._get_num_edge_categories()

    def forward(self, data):
        x = [
            torch.cat(
                [
                    F.one_hot(row[i], num_classes=num)
                    for i, num in enumerate(self.num_node_categories)
                ]
            )
            for row in data.x
        ]

        edge_attr = [
            torch.cat(
                [
                    F.one_hot(row[i], num_classes=num)
                    for i, num in enumerate(self.num_edge_categories)
                ]
            )
            for row in data.edge_attr
        ]

        data.x = torch.stack(x).to(torch.float)
        data.edge_attr = torch.stack(edge_attr).to(torch.float)

        return data

    def get_feature_embedding_dim(self):
        return sum(self._get_num_node_categories())

    def get_edge_embedding_dim(self):
        return sum(self._get_num_edge_categories())

    @staticmethod
    def _get_num_node_categories() -> list[int]:
        return [
            len(pyg_smiles.x_map[prop]) for prop in pyg_smiles.x_map
        ]  # [119, 9, 11, 12, 9, 5, 8, 2, 2]

    @staticmethod
    def _get_num_edge_categories() -> list[int]:
        return [len(pyg_smiles.e_map[prop]) for prop in pyg_smiles.e_map]  # [22, 6, 2]


class CategoricalEncodingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_embedding = CategoricalEmbeddingModel(category_type="node")
        self.edge_embedding = CategoricalEmbeddingModel(category_type="edge")

    def forward(self, data):
        data.x = self.node_embedding(data.x)
        data.edge_attr = self.edge_embedding(data.edge_attr)

        return data

    def get_feature_embedding_dim(self):
        return self.node_embedding.get_node_feature_dim()

    def get_edge_embedding_dim(self):
        return self.edge_embedding.get_edge_feature_dim()


class CategoricalEmbeddingModel(nn.Module):
    """
    Model to embed categorical node or edge features
    """

    def __init__(self, category_type, embedding_dim=8):
        super().__init__()
        if category_type == "node":
            num_categories = self._get_num_node_categories()
        elif category_type == "edge":
            num_categories = self._get_num_edge_categories()
        else:
            print("Invalid category type")
            sys.exit()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_categories[i], embedding_dim)
                for i in range(len(num_categories))
            ]
        )

    def forward(self, x):
        embedded_vars = [
            self.embeddings[i](x[:, i]) for i in range(len(self.embeddings))
        ]

        return torch.cat(embedded_vars, dim=-1)

    def get_node_feature_dim(self):
        return len(self._get_num_node_categories() * self.embedding_dim)

    def get_edge_feature_dim(self):
        return len(self._get_num_edge_categories() * self.embedding_dim)

    @staticmethod
    def _get_num_node_categories() -> list[int]:
        return [
            len(pyg_smiles.x_map[prop]) for prop in pyg_smiles.x_map
        ]  # [119, 9, 11, 12, 9, 5, 8, 2, 2]

    @staticmethod
    def _get_num_edge_categories() -> list[int]:
        return [len(pyg_smiles.e_map[prop]) for prop in pyg_smiles.e_map]  # [22, 6, 2]


class GIN(nn.Module):
    def __init__(self, dim_in, dim_h, edge_dim, dropout=0.1):
        super().__init__()
        self.conv1 = GINEConv(mlp(dim_in, dim_h), edge_dim=edge_dim)
        self.conv2 = GINEConv(mlp(dim_h, dim_h), edge_dim=edge_dim)
        self.conv3 = GINEConv(mlp(dim_h, dim_h), edge_dim=edge_dim)

        self.pool = global_add_pool
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        h = self.conv2(x=h, edge_index=edge_index, edge_attr=edge_attr)
        h = self.conv3(x=h, edge_index=edge_index, edge_attr=edge_attr)

        h_G = self.pool(x=h, batch=data.batch)
        h_G = F.dropout(h_G, p=self.dropout, training=self.training)
        h_G = F.normalize(h_G, dim=1)

        return h_G


class ExtendedConnectivityFingerprintModel:
    def __init__(self):
        self.fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)

    def __call__(self, dataloader):
        mols = [Chem.MolFromSmiles(smiles) for smiles in dataloader.smiles]
        ecfps = [list(ecfp) for ecfp in self.fpgen.GetFingerprints(mols)]
        return ecfps


class ProjectionHead(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, data):
        return self.projection(data)


class FinetuneModel(nn.Module):
    def __init__(self, pretrain_model, out_dim):
        super().__init__()
        self.pretrain_model = pretrain_model
        self.projection_head = ProjectionHead(out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        h_G = self.pretrain_model(data)
        z = self.projection_head(h_G)
        return z
