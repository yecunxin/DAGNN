import csv
import json
import os.path as osp
import numpy as np
import math

import torch
from sympy.physics.quantum.identitysearch import scipy
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.io import read_txt_array
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_undirected
from sklearn.metrics import f1_score
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp
from torch_geometric.utils import to_dense_adj
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)


def evaluate(data, model, conv_time=30):
    model.eval()
    output = model(data.x, data.edge_index, conv_time)

    output = F.log_softmax(output, dim=1)
    loss = F.nll_loss(output, data.y)
    pred = output.max(dim=1)[1]

    correct = pred.eq(data.y).sum().item()
    acc = correct * 1.0 / len(data.y)

    pred = pred.cpu().numpy()
    gt = data.y.cpu().numpy()
    macro_f1 = f1_score(gt, pred, average='macro')
    micro_f1 = f1_score(gt, pred, average='micro')

    return acc, macro_f1, micro_f1, loss


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)),
                                       int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)),
                                       int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def MMD(source_feat, target_feat, sampling_num=1000, times=5):
    source_num = source_feat.size(0)
    target_num = target_feat.size(0)

    source_sample = torch.randint(source_num, (times, sampling_num))
    target_sample = torch.randint(target_num, (times, sampling_num))

    mmd = 0
    for i in range(times):
        source_sample_feat = source_feat[source_sample[i]]
        target_sample_feat = target_feat[target_sample[i]]

        mmd = mmd + get_MMD(source_sample_feat, target_sample_feat)

    mmd = mmd / times
    return mmd


def get_MMD(source_feat, target_feat, kernel_mul=2.0, kernel_num=5
            , fix_sigma=None):
    kernels = guassian_kernel(source_feat,
                              target_feat,
                              kernel_mul=kernel_mul,
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)

    batch_size = min(int(source_feat.size()[0]), int(target_feat.size()[0]))

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def get_katz_matrix(dataset, beta: float = 0.1, method="inverse", max_steps=10):
    A_tilde = to_dense_adj(dataset.edge_index)[0]
    num_nodes = A_tilde.shape[0]
    device = A_tilde.device
    I = torch.eye(num_nodes, device=device)

    if method == "inverse":
        try:
            # 公式：Katz = (I - β*A_tilde)⁻¹ - I
            M = I - beta * A_tilde
            inv_M = torch.linalg.inv(M)
            katz_matrix = inv_M - I
        except torch.linalg.LinAlgError:
            # 矩阵不可逆时切换到级数求和
            print(f"Katz 矩阵求逆失败，自动切换为级数求和（max_steps={max_steps}）")
            method = "series"

    if method == "series":
        # 公式：Katz = Σβ^(k-1)*A^k （k从1到max_steps）
        katz_matrix = torch.zeros_like(A_tilde)
        current_Ak = A_tilde.clone()  # A^1
        for k in range(1, max_steps + 1):
            term = (beta ** (k - 1)) * current_Ak
            katz_matrix += term
            current_Ak = current_Ak @ A_tilde  # 递推计算 A^(k+1)

    return katz_matrix


def get_katz_weight(dataset, beta: float = 0.1, method="inverse", max_steps=10):
    katz_matrix = get_katz_matrix(dataset, beta, method, max_steps)

    non_zero_min = katz_matrix[katz_matrix != 0].min() if (katz_matrix != 0).any() else 1e-8
    katz_matrix[katz_matrix == 0] = non_zero_min
    katz_matrix = torch.log(1 + 1 / katz_matrix)
    katz_weight = katz_matrix / katz_matrix.sum(1).unsqueeze(1) * katz_matrix.shape[0]

    return katz_weight


def SMMD(source_feat, target_feat, katz_weight, sampling_num=1000, times=5):
    source_num = source_feat.size(0)
    target_num = target_feat.size(0)

    if katz_weight.ndim != 2 or katz_weight.size(0) != katz_weight.size(1):
        raise ValueError(f"ppr_weight必须是方阵，实际形状为{katz_weight.shape}")
    if katz_weight.size(0) != target_num:
        raise ValueError(f"ppr_weight维度应为[{target_num}, {target_num}]（目标域大小），实际为{katz_weight.shape}")

    device = source_feat.device
    source_sample = torch.randint(0, source_num, (times, sampling_num), device=device)
    target_sample = torch.randint(0, target_num, (times, sampling_num), device=device)

    smmd = 0.0
    for i in range(times):
        src_idx = source_sample[i]
        tgt_idx = target_sample[i]  # 目标域采样索引

        # 提取采样特征
        source_sample_feat = source_feat[src_idx]
        target_sample_feat = target_feat[tgt_idx]

        # 关键修正：用目标域索引切片目标域的ppr_weight
        sampled_ppr = katz_weight[tgt_idx][:, tgt_idx]  # 形状为[sampling_num, sampling_num]

        smmd += get_SMMD(source_sample_feat, target_sample_feat, sampled_ppr)

    return smmd / times


def get_SMMD(source_feat, target_feat, katz_weight, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    kernels = guassian_kernel(source_feat,
                              target_feat,
                              kernel_mul=kernel_mul,
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)

    batch_size = min(source_feat.size(0), target_feat.size(0))

    if katz_weight.shape != (batch_size, batch_size):
        raise ValueError(f"采样后ppr_weight形状应为[{batch_size}, {batch_size}]，实际为{katz_weight.shape}")

    XX = kernels[:batch_size, :batch_size] * katz_weight
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]

    YY = YY[:batch_size, :batch_size]

    loss = torch.mean(XX + YY - XY - YX)
    return loss


class CitationDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.name = name
        self.root = root
        super(CitationDataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["docs.txt", "edgelist.txt", "labels.txt"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        edge_path = osp.join(self.raw_dir, '{}_edgelist.txt'.format(self.name))
        edge_index = read_txt_array(edge_path, sep=',', dtype=torch.long).t()

        docs_path = osp.join(self.raw_dir, '{}_docs.txt'.format(self.name))
        f = open(docs_path, 'rb')
        content_list = []
        for line in f.readlines():
            line = str(line, encoding="utf-8")
            content_list.append(line.split(","))
        x = np.array(content_list, dtype=float)
        x = torch.from_numpy(x).to(torch.float)

        label_path = osp.join(self.raw_dir, '{}_labels.txt'.format(self.name))
        f = open(label_path, 'rb')
        content_list = []
        for line in f.readlines():
            line = str(line, encoding="utf-8")
            line = line.replace("\r", "").replace("\n", "")
            content_list.append(line)
        y = np.array(content_list, dtype=int)
        y = torch.from_numpy(y).to(torch.int64)

        data_list = []
        data = Data(edge_index=edge_index, x=x, y=y)

        random_node_indices = np.random.permutation(y.shape[0])
        training_size = int(len(random_node_indices) * 0.8)
        val_size = int(len(random_node_indices) * 0.1)
        train_node_indices = random_node_indices[:training_size]
        val_node_indices = random_node_indices[training_size:training_size + val_size]
        test_node_indices = random_node_indices[training_size + val_size:]

        train_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        train_masks[train_node_indices] = 1
        val_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        val_masks[val_node_indices] = 1
        test_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        test_masks[test_node_indices] = 1

        data.train_mask = train_masks
        data.val_mask = val_masks
        data.test_mask = test_masks

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data_list.append(data)

        data, slices = self.collate([data])

        torch.save((data, slices), self.processed_paths[0])


class TwitchDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.name = name
        self.root = root
        super(TwitchDataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["edges.csv, features.json, target.csv"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def load_twitch(self, lang):
        assert lang in ('DE', 'EN', 'FR'), 'Invalid dataset'
        filepath = self.raw_dir
        label = []
        node_ids = []
        src = []
        targ = []
        uniq_ids = set()
        with open(f"{filepath}/musae_{lang}_target.csv", 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                node_id = int(row[5])
                # handle FR case of non-unique rows
                if node_id not in uniq_ids:
                    uniq_ids.add(node_id)
                    label.append(int(row[2] == "True"))
                    node_ids.append(int(row[5]))

        node_ids = np.array(node_ids, dtype=np.int)

        with open(f"{filepath}/musae_{lang}_edges.csv", 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                src.append(int(row[0]))
                targ.append(int(row[1]))

        with open(f"{filepath}/musae_{lang}_features.json", 'r') as f:
            j = json.load(f)

        src = np.array(src)
        targ = np.array(targ)
        label = np.array(label)

        inv_node_ids = {node_id: idx for (idx, node_id) in enumerate(node_ids)}
        reorder_node_ids = np.zeros_like(node_ids)
        for i in range(label.shape[0]):
            reorder_node_ids[i] = inv_node_ids[i]

        n = label.shape[0]
        A = scipy.sparse.csr_matrix((np.ones(len(src)), (np.array(src), np.array(targ))), shape=(n, n))
        features = np.zeros((n, 3170))
        for node, feats in j.items():
            if int(node) >= n:
                continue
            features[int(node), np.array(feats, dtype=int)] = 1
        new_label = label[reorder_node_ids]
        label = new_label

        return A, label, features

    def process(self):
        A, label, features = self.load_twitch(self.name)
        edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
        features = np.array(features)
        x = torch.from_numpy(features).to(torch.float)
        y = torch.from_numpy(label).to(torch.int64)

        data_list = []
        data = Data(edge_index=edge_index, x=x, y=y)

        random_node_indices = np.random.permutation(y.shape[0])
        training_size = int(len(random_node_indices) * 0.8)
        val_size = int(len(random_node_indices) * 0.1)
        train_node_indices = random_node_indices[:training_size]
        val_node_indices = random_node_indices[training_size:training_size + val_size]
        test_node_indices = random_node_indices[training_size + val_size:]

        train_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        train_masks[train_node_indices] = 1
        val_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        val_masks[val_node_indices] = 1
        test_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        test_masks[test_node_indices] = 1

        data.train_mask = train_masks
        data.val_mask = val_masks
        data.test_mask = test_masks

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data_list.append(data)

        data, slices = self.collate([data])

        torch.save((data, slices), self.processed_paths[0])


class Writer(object):
    def __init__(self, path):
        self.writer = SummaryWriter(path)

    def scalar_logger(self, tag, value, step):
        """Log a scalar variable."""
        # if self.local_rank == 0:
        self.writer.add_scalar(tag, value, step)

    def scalars_logger(self, tag, value, step):
        """Log a scalar variable."""
        # if self.local_rank == 0:
        self.writer.add_scalars(tag, value, step)

    def image_logger(self, tag, images, step):
        """Log a list of images."""
        # if self.local_rank == 0:
        self.writer.add_image(tag, images, step)

    def histo_logger(self, tag, values, step):
        """Log a histogram of the tensor of values."""
        # if self.local_rank == 0:
        self.writer.add_histogram(tag, values, step, bins='auto')




def compute_similarity(
        features: torch.Tensor,
        topk: int = 15,
        normalize: bool = True
) -> torch.Tensor:

    N = features.size(0)
    if N == 0:
        return torch.empty((2, 0), dtype=torch.long, device=features.device)

    if normalize:
        features = F.normalize(features, p=2, dim=1)  # [N, D]

    sim_matrix = torch.matmul(features, features.T)  # [N, N]
    sim_matrix.fill_diagonal_(0.0)

    topk = min(topk, N - 1)
    if topk <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=features.device)

    _, topk_indices = torch.topk(sim_matrix, k=topk, dim=1)  # [N, topk]

    row_indices = torch.arange(N, device=features.device).unsqueeze(1).repeat(1, topk).flatten()  # [N×topk]

    col_indices = topk_indices.flatten()  # [N×topk]

    edge_index = torch.stack([row_indices, col_indices], dim=0)  # [2, N×topk]
    edge_index = edge_index.unique(dim=1)

    assert edge_index.shape[0] == 2, f"边索引形状错误：{edge_index.shape}"
    return edge_index.long()


def structure_aware_regularization(
        y_hat: torch.Tensor,
        edge_index: torch.Tensor,
        batch_size: int = 1024,
        normalize_y: bool = True
) -> torch.Tensor:
    device = y_hat.device
    N = y_hat.size(0)
    if N == 0:
        return torch.tensor(0.0, device=device)

    if normalize_y:
        y_hat = F.normalize(y_hat, p=2, dim=1)

    adj = to_dense_adj(edge_index, max_num_nodes=N).squeeze(0).to(device)
    if adj.shape != (N, N):
        adj = torch.zeros(N, N, device=device)
    adj.fill_diagonal_(0)

    total_loss = torch.tensor(0.0, device=device)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_size_current = end - start
        batch_indices = slice(start, end)

        y_batch = y_hat[batch_indices]
        adj_batch = adj[batch_indices]

        sim = torch.matmul(y_batch, y_hat.t())
        sigma_sim = torch.sigmoid(sim).clamp(1e-8, 1 - 1e-8)
        log_sigma = torch.log(sigma_sim)
        log_1_minus_sigma = torch.log(1 - sigma_sim)

        assert adj_batch.shape == (batch_size_current, N), f"adj_batch形状错误：{adj_batch.shape}"
        assert log_sigma.shape == (batch_size_current, N), f"log_sigma形状错误：{log_sigma.shape}"

        self_mask = torch.zeros(batch_size_current, N, dtype=torch.bool, device=device)
        self_mask[:, start:end] = torch.eye(batch_size_current, device=device, dtype=torch.bool)
        loss_matrix = adj_batch * log_sigma + (1 - adj_batch) * log_1_minus_sigma
        batch_loss = -loss_matrix[~self_mask].sum()
        total_loss += batch_loss


    num_pairs = N * (N - 1)
    return total_loss / num_pairs if num_pairs > 0 else total_loss