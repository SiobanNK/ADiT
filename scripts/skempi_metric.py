import pickle as pkl
import torch
from torcheval.metrics import BinaryAUROC

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from adit.common.variadic_utils import spearmanr, pearsonr

import argparse


def mse(preds, targets):
    return torch.nn.functional.mse_loss(preds, targets)

def mae(preds, targets):
    return torch.nn.functional.l1_loss(preds, targets)

def load_pkl(fpaths: list[str]):
    accession_codes = []
    all_pred = []
    all_target = []
    for fpath in fpaths:
        with open(fpath, "rb") as f:
            accession_code, pred, target = pkl.load(f)
        accession_codes += accession_code
        all_pred.append(pred.cpu())
        all_target.append(target.cpu())
    all_pred = torch.concat(all_pred, dim=-1)
    all_target = torch.concat(all_target, dim=-1)
    return accession_codes, all_pred, all_target


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str)
args = parser.parse_known_args()[0]

if __name__ == "__main__":
    input_dir = os.path.expanduser(args.input_dir)
    result_paths = [ # example
        os.path.join(input_dir, "result_split_0.pkl"),
        os.path.join(input_dir, "result_split_1.pkl"),
        os.path.join(input_dir, "result_split_2.pkl"),
    ]
    accession_codes, all_pred, all_target = load_pkl(result_paths)
    metrics = {}
    metrics["overall_pearsonr"] = pearsonr(all_pred, all_target)
    metrics["overall_spearmanr"] = spearmanr(all_pred, all_target)
    metrics["overall_rmse"] = mse(all_pred, all_target) ** 0.5
    metrics["overall_mae"] = mae(all_pred, all_target)
    print(metrics)
