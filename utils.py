import csv
import json
import os.path as osp
import numpy as np
import math
import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.io import read_txt_array
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import f1_score


def evaluate(data, encoder, classifier, conv_time=0, is_source=True):
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        if is_source:
            _, z_shared, _ = encoder(data.x, data.edge_index, conv_time=conv_time, is_source=True)
        else:
            _, z_shared, _, _ = encoder(data.x, data.edge_index, conv_time=conv_time, is_source=False)
        output = classifier(z_shared)
        output = F.log_softmax(output, dim=1)
        pred = output.max(dim=1)[1]
        correct = pred.eq(data.y).sum().item()
        acc = correct * 1.0 / len(data.y)
        macro_f1 = f1_score(data.y.cpu().numpy(), pred.cpu().numpy(), average='macro')
        micro_f1 = f1_score(data.y.cpu().numpy(), pred.cpu().numpy(), average='micro')
    return acc, macro_f1, micro_f1


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def batch_hsic(x, y, batch_size=256):
    N, D = x.size()
    device = x.device

    def compute_gaussian_bandwidth(feat):
        sample_size = min(10000, N * N)
        idx1 = torch.randint(0, N, (sample_size,), device=device)
        idx2 = torch.randint(0, N, (sample_size,), device=device)
        dist = torch.norm(feat[idx1] - feat[idx2], p=2, dim=1)
        sigma = torch.median(dist) + 1e-6
        return sigma

    sigma_x = compute_gaussian_bandwidth(x)
    sigma_y = compute_gaussian_bandwidth(y)
    num_batches = math.ceil(N / batch_size)
    tr_KxKy = 0.0
    tr_Kx = 0.0
    tr_Ky = 0.0

    for b in range(num_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, N)
        x_batch = x[start:end]
        y_batch = y[start:end]

        def gaussian_kernel_batch_full(feat_batch, feat_full, sigma):
            dist = torch.cdist(feat_batch, feat_full, p=2)
            return torch.exp(-dist ** 2 / (2 * sigma ** 2))

        Kx_batch = gaussian_kernel_batch_full(x_batch, x, sigma_x)
        Ky_batch = gaussian_kernel_batch_full(y_batch, y, sigma_y)

        batch_tr_KxKy = (Kx_batch * Ky_batch).sum()
        tr_KxKy += batch_tr_KxKy

        batch_idx = torch.arange(start, end, device=device) - start
        full_idx = torch.arange(start, end, device=device)
        tr_Kx += Kx_batch[batch_idx, full_idx].sum()
        tr_Ky += Ky_batch[batch_idx, full_idx].sum()

    term1 = tr_KxKy
    term2 = 2 * tr_Kx * tr_Ky / N
    term3 = tr_Kx * tr_Ky / N
    tr_RKxRKy = term1 - term2 + term3
    hsic = tr_RKxRKy / ((N - 1) ** 2)
    return hsic


def compute_ppr_matrix(edge_index, num_nodes, alpha=0.1, tol=1e-6, max_steps=100):
    A_tilde = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    device = A_tilde.device
    I = torch.eye(num_nodes, device=device)
    A_sum = A_tilde.sum(dim=1, keepdim=True).clamp(min=1e-8)
    A_norm = A_tilde / A_sum
    ppr_matrix = torch.zeros_like(A_tilde)

    for i in range(num_nodes):
        r = torch.zeros(num_nodes, device=device)
        r[i] = 1.0
        prev_r = r.clone()
        for _ in range(max_steps):
            r = alpha * I[i] + (1 - alpha) * torch.matmul(A_norm.T, r)
            if torch.norm(r - prev_r, p=1) < tol:
                break
            prev_r = r.clone()
        ppr_matrix[i] = r
    ppr_matrix.fill_diagonal_(0.0)
    return ppr_matrix


def compute_tmmd(source_feat, target_feat, ppr_matrix, sampling_num=1000, times=5, kernel_mul=2.0, kernel_num=5):
    source_num = source_feat.size(0)
    target_num = target_feat.size(0)
    device = source_feat.device

    epsilon = 1e-8
    gamma = torch.where(ppr_matrix == 0, torch.log(1 + 1/epsilon), torch.log(1 + 1/ppr_matrix))
    gamma = gamma.to(device)

    source_sample = torch.randint(0, source_num, (times, sampling_num), device=device)
    target_sample = torch.randint(0, target_num, (times, sampling_num), device=device)
    tmmd_total = 0.0

    for i in range(times):
        src_idx = source_sample[i]
        tgt_idx = target_sample[i]
        src_feat = source_feat[src_idx]
        tgt_feat = target_feat[tgt_idx]

        kernels = gaussian_kernel(src_feat, tgt_feat, kernel_mul, kernel_num)
        batch_size = min(src_feat.size(0), tgt_feat.size(0))

        gamma_sampled = gamma[tgt_idx][:, tgt_idx]
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:] * gamma_sampled
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]

        YY = YY[:batch_size, :batch_size]
        term1 = torch.sum(YY) / torch.sum(gamma_sampled)
        term2 = -2 * torch.mean(XY)
        term3 = torch.mean(XX)
        tmmd_total += (term1 + term2 + term3)

    return tmmd_total / times



def feature_l2_recon_loss(x_origin, x_recon):
    feature_diff = x_origin - x_recon
    node_l2_norms = torch.norm(feature_diff, p=2, dim=1)
    loss = torch.mean(node_l2_norms)
    return loss

def scaled_cosine_loss(x_origin, x_recon):
    x_origin_norm = F.normalize(x_origin, p=2, dim=1)
    x_recon_norm = F.normalize(x_recon, p=2, dim=1)
    cos_sim = torch.sum(x_origin_norm * x_recon_norm, dim=1)
    loss = torch.mean(1 - cos_sim)
    return loss


def adj_bce_loss(adj_origin, adj_recon):
    loss = F.binary_cross_entropy(adj_recon, adj_origin)
    return loss


class CitationDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        self.root = root
        super(CitationDataset, self).__init__(root, transform, pre_transform)
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
        with open(docs_path, 'r', encoding='utf-8') as f:
            content_list = [line.strip().split(",") for line in f.readlines()]
        x = np.array(content_list, dtype=float)
        x = torch.from_numpy(x).to(torch.float)

        label_path = osp.join(self.raw_dir, '{}_labels.txt'.format(self.name))
        with open(label_path, 'r', encoding='utf-8') as f:
            content_list = [line.strip().replace("\r", "").replace("\n", "") for line in f.readlines()]
        y = np.array(content_list, dtype=int)
        y = torch.from_numpy(y).to(torch.int64)

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

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])


class TwitchDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        self.root = root
        super(TwitchDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["edges.csv", "features.json", "target.csv"]

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
                if node_id not in uniq_ids:
                    uniq_ids.add(node_id)
                    label.append(int(row[2] == "True"))
                    node_ids.append(int(row[5]))
        node_ids = np.array(node_ids, dtype=int)

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
        A = np.zeros((n, n))
        for s, t in zip(src, targ):
            if s < n and t < n:
                A[s, t] = 1
                A[t, s] = 1
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
        x = torch.from_numpy(features).to(torch.float)
        y = torch.from_numpy(label).to(torch.int64)

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

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
