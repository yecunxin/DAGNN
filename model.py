import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import zeros_, glorot
from layers import AsymmetricGCNConv


class AsymmetricDecoupleEncoder(torch.nn.Module):
    def __init__(self, args):
        super(AsymmetricDecoupleEncoder, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.dropout_ratio = args.dropout_ratio

        self.private_encoder_source = AsymmetricGCNConv(self.num_features, self.nhid)
        self.private_encoder_target = AsymmetricGCNConv(self.num_features, self.nhid)

        self.shared_encoder = AsymmetricGCNConv(self.num_features, self.nhid)

        self.decoder_shared_source = nn.Sequential(
            nn.Linear(2 * self.nhid, self.nhid),
            nn.ReLU(),
            nn.Linear(self.nhid, self.num_features)
        )
        self.decoder_shared_target = nn.Sequential(
            nn.Linear(2 * self.nhid, self.nhid),
            nn.ReLU(),
            nn.Linear(self.nhid, self.num_features)
        )

    def forward(self, x, edge_index, conv_time=0, is_source=True):
        if is_source:
            z_private = self.private_encoder_source(x, edge_index, conv_time=conv_time)
            z_shared = self.shared_encoder(x, edge_index, conv_time=conv_time)
        else:
            z_private = self.private_encoder_target(x, edge_index, conv_time=conv_time)
            z_shared = self.shared_encoder(x, edge_index, conv_time=conv_time)

        z_private = F.relu(z_private)
        z_shared = F.relu(z_shared)
        z_private = F.dropout(z_private, p=self.dropout_ratio, training=self.training)
        z_shared = F.dropout(z_shared, p=self.dropout_ratio, training=self.training)

        z_concat = torch.cat([z_private, z_shared], dim=1)

        if is_source:
            x_recon = self.decoder_shared_source(z_concat)
            return z_private, z_shared, x_recon
        else:
            x_recon = self.decoder_shared_target(z_concat)
            adj_recon = torch.sigmoid(torch.matmul(x_recon, x_recon.T))
            return z_private, z_shared, x_recon, adj_recon


class Classifier(torch.nn.Module):
    def __init__(self, in_dim, num_classes):
        super(Classifier, self).__init__()
        self.lin = nn.Linear(in_dim, num_classes)
        glorot(self.lin.weight)
        zeros_(self.lin.bias)

    def forward(self, x):
        return self.lin(x)
