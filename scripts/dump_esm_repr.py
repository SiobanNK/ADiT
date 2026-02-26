import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__name__)))
import pickle as pkl
import torch
import numpy as np
from glob import glob
from adit.models.net.esm2 import esm
import tqdm


def access_code(fpath):
    return os.path.splitext(os.path.basename(fpath))[0]


def process_one(fpath, device, prefix = 'wt'):
    assert prefix in ['wt', 'mt']
    with open(fpath, 'rb') as f:
        data_object = pkl.load(f)

    chain_index = torch.as_tensor(data_object[f'{prefix}_data_object']['chain_index'])
    starts = torch.cat(
        [torch.BoolTensor([True]), chain_index[1:] != chain_index[:-1]], dim = 0
    )
    starts = torch.where(starts)[0]
    ends = torch.cat(
        [starts[1:], torch.LongTensor([chain_index.shape[0]])], dim = 0
    )
    aatype = torch.as_tensor(data_object[f'{prefix}_data_object']['aatype'])
    num_residues_per_chain = ends - starts

    return aatype.to(device), num_residues_per_chain.to(device)


def process_one_antibody(fpath, device):
    with open(fpath, 'rb') as f:
        data_object = pkl.load(f)

    chain_index = torch.as_tensor(data_object['chain_index'])
    starts = torch.cat(
        [torch.BoolTensor([True]), chain_index[1:] != chain_index[:-1]], dim = 0
    )
    starts = torch.where(starts)[0]
    ends = torch.cat(
        [starts[1:], torch.LongTensor([chain_index.shape[0]])], dim = 0
    )
    aatype = torch.as_tensor(data_object['aatype'])
    num_residues_per_chain = ends - starts

    return aatype.to(device), num_residues_per_chain.to(device)

@torch.no_grad()
def main_skempi():
    skempi_dir_path = "dataset/skempi"
    device = 'cuda:0'

    data_path = []
    for split_id in range(3):
        data_path = data_path + glob(
            os.path.join(skempi_dir_path, f'split_{split_id}', '*')
        )
    
    esm_weight_path = 'ckpts'
    esm_model = 'ESM-2-650M'
    esm_encoder = esm.ESM(path = esm_weight_path, model = esm_model).to(device)

    save_dir = "dataset/skempi_esm_repr"
    os.makedirs(save_dir, exist_ok=True)
    for fpath in tqdm.tqdm(data_path):
        a_code = access_code(fpath)

        for prefix in ['wt', 'mt']:
            save_path = os.path.join(save_dir, f"{prefix}_{a_code}.pt")
            if os.path.exists(save_path): 
                continue

            aatype, num_residues_per_chain = process_one(fpath, device, prefix)
            if '2NZ9' in fpath or '2NYY' in fpath:
                num_residues_per_chain = torch.LongTensor([
                    400, num_residues_per_chain[0] - 400, 
                    num_residues_per_chain[1], num_residues_per_chain[2]
                ]).to(device)

            esm_repr = esm_encoder(aatype, num_residues_per_chain)
            torch.save(esm_repr.cpu(), save_path)

@torch.no_grad()
def main_HER2():
    device = 'cuda:0'
    data_path = []

    # HER2
    data_path = glob(
        os.path.join('dataset/HER2', 'processed', '*')
    )
    
    esm_weight_path = 'ckpts'
    esm_model = 'ESM-2-650M'
    esm_encoder = esm.ESM(path = esm_weight_path, model = esm_model).to(device)
    
    save_dir = "dataset/HER2_esm_repr"
    os.makedirs(save_dir, exist_ok=True)
    for fpath in tqdm.tqdm(data_path):
        a_code = access_code(fpath)
        save_path = os.path.join(save_dir, "%s.pt" % a_code)
        if not os.path.exists(save_path):
            aatype, num_residues_per_chain = process_one_antibody(fpath, device)
            esm_repr = esm_encoder(aatype, num_residues_per_chain)
            torch.save(esm_repr.cpu(), save_path)


if __name__ == "__main__":
    main_skempi()
    main_HER2()
