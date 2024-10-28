import sys
import torch
import torch.nn.functional as F
import torch_geometric.utils.smiles as pyg_smiles
from torch import nn
from torch_geometric.nn import GINEConv, global_add_pool


def mlp(dim_in, dim_out):
    return nn.Sequential(
        nn.Linear(dim_in, dim_out),
        nn.BatchNorm1d(dim_out),
        nn.ReLU(),
        nn.Linear(dim_out, dim_out),
        nn.ReLU(),
    )


class CategoricalEmbeddingModel(nn.Module):
    """
    Model to embed categorical node or edge features
    """

    def __init__(self, category_type, embedding_dim=8):
        super().__init__()
        if category_type == 'node':
            num_categories = self._get_num_node_categories()
        elif category_type == 'edge':
            num_categories = self._get_num_edge_categories()
        else:
            print('Invalid category type')
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
        return [len(pyg_smiles.x_map[prop]) for prop in pyg_smiles.x_map] # [119, 9, 11, 12, 9, 5, 8, 2, 2]

    @staticmethod
    def _get_num_edge_categories() -> list[int]:
        return [len(pyg_smiles.e_map[prop]) for prop in pyg_smiles.e_map] # [22, 6, 2]


class GIN(nn.Module):
    def __init__(self, dim_h, dim_in, edge_dim, dropout=0.1):
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
