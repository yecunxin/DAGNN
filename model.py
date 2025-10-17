import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GCNConv, AttentionDepthGCNConv


class Encoder(torch.nn.Module):

    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.dropout_ratio = args.dropout_ratio

        self.attConv = AttentionDepthGCNConv(self.num_features, self.nhid)

    def forward(self, x, edge_index, conv_time=30):

        x = self.attConv(x, edge_index, conv_time)

        x = F.relu(x)

        return x


class Classifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Classifier, self).__init__()
        self.cls = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.cls(x, edge_index)
        return x