import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import NNConv
from torch_geometric.nn.pool import global_mean_pool


class mpnn_layer(nn.Module):
    """
    MPNN from
    `Neural Message Passing for Quantum Chemistry <https://arxiv.org/abs/1704.01212>`
    Parameters
    ----------
    node_input_dim : int
        Dimension of input node feature, default to be 42.
    edge_input_dim : int
        Dimension of input edge feature, default to be 10.
    node_hidden_dim : int
        Dimension of node feature in hidden layers, default to be 42.
    edge_hidden_dim : int
        Dimension of edge feature in hidden layers, default to be 128.
    num_step_message_passing : int
        Number of message passing steps, default to be 6.
    """

    def __init__(self,
                 node_input_dim=39,
                 edge_input_dim=10,
                 node_hidden_dim=39,
                 edge_hidden_dim=39,
                 num_step_message_passing=5,
                 dropout=0.0,
                 ):
        super().__init__()
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        # self.set2set = Set2Set(node_hidden_dim, 2, 1)
        # self.set2set = Set2Set(node_hidden_dim, 2, num_layers=1)
        self.message_layer = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim), nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim))
        # self.conv = nn.ModuleList(NNConv(in_channels=node_hidden_dim,
        #                    out_channels=node_hidden_dim,
        #                    nn=edge_network,
        #                    aggr='add',
        #                    root_weight=True
        #                    ) for _ in range(num_step_message_passing))
        self.conv = NNConv(in_channels=node_hidden_dim,
                           out_channels=node_hidden_dim,
                           nn=edge_network,
                           aggr='add',
                           root_weight=True
                           )
        self.lns = nn.ModuleList([nn.LayerNorm(node_hidden_dim) for _ in range(num_step_message_passing)])

        self.dropout = dropout

    def forward(self, g):
        """Returns the node embeddings after message passing phase.
        Parameters
        ----------
        g : Torch geometric batch data
            Input batch data for molecule(s)
        Returns
        -------
        res : node features
        """

        x, edge_index, edge_attr, batch_index = g.x.float(), g.edge_index, g.edge_attr.float(), g.batch

        out= F.relu(self.lin0(x))

        for i in range(self.num_step_message_passing):
            if len(g.edge_attr) != 0:
                m = torch.relu(self.conv(out, edge_index, edge_attr))
            else:
                m = torch.relu(self.conv.bias + out)
            out = self.message_layer(torch.cat([m, out], dim=1))
            out = self.lns[i](out)
            if i != self.num_step_message_passing-1:
                out = F.relu(out)
                out = F.dropout(out, p=self.dropout, training=self.training)

        return out

class MLP(nn.Module):
    def __init__(self, d_in_feats, d_out_feats, n_dense_layers, activation, d_hidden_feats=None):
        super(MLP, self).__init__()
        self.n_dense_layers = n_dense_layers
        self.d_hidden_feats = d_out_feats if d_hidden_feats is None else d_hidden_feats
        self.dense_layer_list = nn.ModuleList()
        self.in_proj = nn.Linear(d_in_feats, self.d_hidden_feats)
        for _ in range(self.n_dense_layers - 2):
            self.dense_layer_list.append(nn.Linear(self.d_hidden_feats, self.d_hidden_feats))
        self.out_proj = nn.Linear(self.d_hidden_feats, d_out_feats)
        self.act = activation

    def forward(self, feats):
        feats = self.act(self.in_proj(feats))
        for i in range(self.n_dense_layers - 2):
            feats = self.act(self.dense_layer_list[i](feats))
        feats = self.out_proj(feats)
        return feats


class MPNN(nn.Module):
    def __init__(self,
                 node_input_dim=46,
                 edge_input_dim=10,
                 node_hidden_dim=256,
                 edge_hidden_dim=256,
                 num_layers=4,
                 num_layer_mlp=3,
                 dropout_ratio=0,
                 model_type='MPNN'
                 ):
        super().__init__()

        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        self.model_type = model_type

        if self.model_type == 'MPNN':
            from MPNN_codes.layers.mpnn_layer import mpnn_layer
            self.mol_gather = mpnn_layer(self.node_input_dim, self.edge_input_dim,
                                   self.node_hidden_dim, self.edge_input_dim,
                                   self.num_layers)

        elif self.model_type == 'GCN':
            from layers.gcn_layer import GCN
            self.mol_gather = GCN(self.node_input_dim, self.node_hidden_dim, self.node_hidden_dim * 2, self.num_layers)

        elif self.model_type == 'GAT':
            from layers.gat_layer import GAT
            self.mol_gather = GAT(self.node_input_dim, self.node_hidden_dim, self.node_hidden_dim * 2, self.num_layers)

        elif self.model_type == 'GIN':
            from layers.gin_layer import GIN
            self.mol_gather = GIN(self.node_input_dim, self.node_hidden_dim, self.node_hidden_dim * 2, self.num_layers)

        elif self.model_type == 'GraphSAGE':
            from layers.graphsage_layer import GraphSAGE
            self.mol_gather = GraphSAGE(self.node_input_dim, self.node_hidden_dim, self.node_hidden_dim * 2, self.num_layers)
        #
        # self.set2set = Set2Set(node_hidden_dim * 2, processing_steps=4, num_layers=2)

        self.predictor = MLP(self.node_hidden_dim, 1 , num_layer_mlp, activation=nn.ReLU(), d_hidden_feats=self.node_hidden_dim)
        # self.init_model()

    # def init_model(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             torch.nn.init.xavier_uniform_(m.weight.data)
    #             if m.bias is not None:
    #                 m.bias.data.fill_(0.0)

    def forward(self, data):

        # node embeddings for each graph
        mol_features = self.mol_gather(data)

        # Add normalization
        # self.mol_features = F.normalize(mol_features, dim=1)

        # readout (pooling) after normalization
        mol_features = global_mean_pool(mol_features, data.batch)
        # self.final_features = self.set2set(self.mol_features, mol.batch)

        # get predictions
        predictions = self.predictor(mol_features)

        return predictions

    def get_checkpoints(self):

        return self.final_features