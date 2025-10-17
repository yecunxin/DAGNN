import time
import glob
import argparse
import itertools
import os
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from model import Encoder, Classifier
from utils import evaluate, CitationDataset, TwitchDataset, structure_aware_regularization, SMMD, get_katz_weight, compute_similarity

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.005,
                    help='weight decay')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--nhid', type=int, default=64,
                    help='hidden size')
parser.add_argument('--patience', type=int, default=100,
                    help='patience for early stopping')
parser.add_argument('--device', type=str, default='cuda',
                    help='specify cuda devices')
parser.add_argument('--run_times', type=int, default=1,
                    help='run times')
parser.add_argument('--epochs', type=int, default=200,
                    help='maximum number of epochs')
parser.add_argument('--source', type=str, default='DBLPv7',
                    help='source domain data')
parser.add_argument('--target', type=str, default='Citationv1',
                    help='target domain data')
parser.add_argument('--source_pnum', type=int, default=0,
                    help='the number of propagation layers on the source graph')
parser.add_argument('--target_pnum', type=int, default=10,
                    help='the number of propagation layers on the target graph')
args = parser.parse_args()
print(args)

if args.source in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data',
                    args.source)
    source_dataset = CitationDataset(path, args.source)
if args.source in {'EN', 'DE'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data',
                    args.source)
    source_dataset = TwitchDataset(path, args.source)
if args.target in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data',
                    args.target)
    target_dataset = CitationDataset(path, args.target)
if args.target in {'EN', 'DE'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data',
                    args.target)
    target_dataset = TwitchDataset(path, args.target)
source_data = source_dataset[0].to(args.device)
target_data = target_dataset[0].to(args.device)

args.num_classes = len(np.unique(source_dataset[0].y.numpy()))
args.num_features = source_data.x.size(1)
args.save_path = './'


def train(args, source_data, target_data):
    min_loss = 1e10
    patience_cnt = 0
    loss_values = []
    best_epoch = 0
    tau = 0.5

    encoder = Encoder(args).to(args.device)

    # 分类器
    cls = Classifier(args.nhid, args.num_classes).to(args.device)

    models = [encoder, cls]
    params = itertools.chain(*[model.parameters() for model in models])

    optimizer = torch.optim.Adam(params, lr=args.lr,
                                 weight_decay=args.weight_decay)

    t = time.time()
    for model in models:
        model.train()

    for epoch in range(args.epochs):
        correct = 0

        # Source Domain Cross-Entropy Loss

        x_s = encoder(source_data.x, source_data.edge_index, args.source_pnum)

        output = cls(x_s, source_data.edge_index)

        train_loss = F.nll_loss(F.log_softmax(output / tau, dim=1), source_data.y)
        loss = train_loss

        # SMMD Loss

        x_t = encoder(target_data.x, target_data.edge_index, args.target_pnum)

        beta = 0.5  # 可调超参数（建议范围：0.01~0.5，需实验调优）
        katz_weight_t = get_katz_weight(
            target_data,  # 目标域数据集（与 PPR 一致）
            beta=beta,
            method="series"  # 优先矩阵求逆，失败自动切换级数求和
        )

        smmd_loss = SMMD(x_s, x_t, katz_weight_t)

        str_loss = structure_aware_regularization(x_t, compute_similarity(x_t))

        # Overall Loss
        loss = 1 * loss + 0.5 * smmd_loss + 0.5 * str_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for model in models:
            model.eval()
        with torch.no_grad():
            # acc, _, _, _ = evaluate(source_data, model)
            # _, macro_f1, micro_f1, test_loss = evaluate(target_data, models,
            #                                             args.target_pnum)

            output_t = cls(x_t, target_data.edge_index)
            output_t = F.log_softmax(output_t, dim=1)
            loss = F.nll_loss(output_t / tau, target_data.y)
            pred = output_t.max(dim=1)[1]

            correct = pred.eq(target_data.y).sum().item()
            acc = correct * 1.0 / len(target_data.y)

            pred = pred.cpu().numpy()
            gt = target_data.y.cpu().numpy()
            macro_f1 = f1_score(gt, pred, average='macro')
            micro_f1 = f1_score(gt, pred, average='micro')

            print('Epoch: {:04d}'.format(epoch + 1),
                  'train_loss: {:.6f}'.format(loss),
                  # 'test_loss: {:.6f}'.format(test_loss),
                  'test_acc: {:.6f}'.format(acc),
                  'macro_f1: {:.6f}'.format(macro_f1),
                  'micro_f1: {:.6f}'.format(micro_f1))

train(args, source_data, target_data)
