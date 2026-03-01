import argparse
import itertools
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from model import AsymmetricDecoupleEncoder, Classifier
from utils import evaluate, CitationDataset, TwitchDataset, batch_hsic, compute_ppr_matrix, compute_tmmd, scaled_cosine_loss, adj_bce_loss

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--dropout_ratio', type=float, default=0.5)
parser.add_argument('--nhid', type=int, default=128)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--run_times', type=int, default=10)
parser.add_argument('--source', type=str, default='DBLPv7')
parser.add_argument('--target', type=str, default='Citationv1')
parser.add_argument('--source_pnum', type=int, default=0)
parser.add_argument('--target_pnum', type=int, default=10)
parser.add_argument('--lambda1', type=float, default=0.5)
parser.add_argument('--lambda2', type=float, default=1e-4)
parser.add_argument('--lambda3', type=float, default=0.1)
parser.add_argument('--ppr_alpha', type=float, default=0.1)
parser.add_argument('--hsic_batch_size', type=int, default=256)
args = parser.parse_args()

if args.source in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data', args.source)
    source_dataset = CitationDataset(path, args.source)
if args.source in {'EN', 'DE'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data', args.source)
    source_dataset = TwitchDataset(path, args.source)
if args.target in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data', args.target)
    target_dataset = CitationDataset(path, args.target)
if args.target in {'EN', 'DE'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data', args.target)
    target_dataset = TwitchDataset(path, args.target)

source_data = source_dataset[0].to(args.device)
target_data = target_dataset[0].to(args.device)
args.num_classes = len(np.unique(source_dataset[0].y.numpy()))
args.num_features = source_data.x.size(1)

target_num_nodes = target_data.x.size(0)
ppr_matrix = compute_ppr_matrix(target_data.edge_index, target_num_nodes, alpha=args.ppr_alpha).to(args.device)

adj_s = torch.zeros(source_data.x.size(0), source_data.x.size(0), device=args.device)
adj_s[source_data.edge_index[0], source_data.edge_index[1]] = 1.0
adj_t = torch.zeros(target_data.x.size(0), target_data.x.size(0), device=args.device)
adj_t[target_data.edge_index[0], target_data.edge_index[1]] = 1.0

def train():
    encoder = AsymmetricDecoupleEncoder(args).to(args.device)
    cls = Classifier(args.nhid, args.num_classes).to(args.device)
    models = [encoder, cls]
    params = itertools.chain(*[model.parameters() for model in models])
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    for epoch in range(args.epochs):
        for model in models:
            model.train()
        optimizer.zero_grad()

        z_private_s, z_shared_s, x_recon_s = encoder(
            source_data.x, source_data.edge_index,
            conv_time=args.source_pnum, is_source=True
        )

        z_private_t, z_shared_t, x_recon_t, adj_recon_t = encoder(
            target_data.x, target_data.edge_index,
            conv_time=args.target_pnum, is_source=False
        )

        output = cls(z_shared_s)
        loss_cls = F.nll_loss(F.log_softmax(output, dim=1), source_data.y)

        hsic_s = batch_hsic(z_private_s, z_shared_s, batch_size=args.hsic_batch_size)
        hsic_t = batch_hsic(z_private_t, z_shared_t, batch_size=args.hsic_batch_size)
        loss_diff = hsic_s + hsic_t

        loss_rec_s = scaled_cosine_loss(source_data.x, x_recon_s)
        loss_rec_t = adj_bce_loss(adj_t, adj_recon_t)
        loss_rec = loss_rec_t + 0.5 * loss_rec_s

        loss_tmmd = compute_tmmd(z_shared_s, z_shared_t, ppr_matrix)

        loss = loss_cls + args.lambda1 * loss_tmmd + args.lambda2 * loss_diff + args.lambda3 * loss_rec

        loss.backward()
        optimizer.step()

        acc, macro_f1, micro_f1 = evaluate(target_data, encoder, cls, conv_time=args.target_pnum, is_source=False)
        if acc > best_acc:
            best_acc = acc
        print(f'Epoch: {epoch+1:04d} | Loss: {loss.item():.6f} | Target Acc: {acc:.4f} | Best Acc: {best_acc:.4f}')

    return best_acc

if __name__ == '__main__':
    acc_list = []
    for run in range(args.run_times):
        print(f'\n===== Run {run+1}/{args.run_times} =====')
        run_acc = train()
        acc_list.append(run_acc)
    print(f'\nFinal Result: Mean Acc = {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}')
