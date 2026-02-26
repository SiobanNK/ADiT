import numpy as np
import torch
from torch_cluster import nearest
import csv
import tree
import pickle as pkl

from adit.common import residue_constants, protein
CA_IDX = residue_constants.atom_order['CA']


def load_pkl(fpath):
    with open(fpath, 'rb') as f:
        data_object = pkl.load(f)
    return data_object


# read csv format file without the first row
def read_csv(fpath):
    rows = []
    with open(fpath, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            rows.append(row)
    
    return rows


def csv2dicts(csv_file_path):
    list_of_dict = []
    with open(csv_file_path, 'r', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            list_of_dict.append(dict(row))
    return list_of_dict


def get_subgraph(data_object, res_mask):
    for k in data_object.keys():
        data_object[k] = data_object[k][res_mask]
    return data_object


def extract_dest_protein_and_mutation_mask(data_object, chain_a, chain_b, mutations):
    entity_a = torch.zeros(data_object["aatype"].shape[0], dtype=torch.bool)
    alphabet2id = data_object["chain_id_mapping"]
    
    for a in chain_a:
        entity_a |= data_object["chain_index"] == alphabet2id[a]

    entity_b = torch.zeros(data_object["aatype"].shape[0], dtype=torch.bool)
    for a in chain_b:
        entity_b |= data_object["chain_index"] == alphabet2id[a]

    is_mutation = torch.zeros(data_object["aatype"].shape[0], dtype=torch.bool)
    residue_number = data_object["original_index"]
    del data_object["original_index"]

    for m in mutations:
        is_mutation |= \
            (data_object["chain_index"] == alphabet2id[m[1]]) & \
            (residue_number == int(m[2:-1]))
    
    res_mask = (entity_a | entity_b).bool()
    del data_object["chain_id_mapping"]
    data_object = get_subgraph(data_object, res_mask)
    is_mutation = is_mutation[res_mask]
    return data_object, is_mutation


def truncation_skempi(data_object, is_mutation, k, consecutive = False):
    k = min(k, data_object["aatype"].shape[0])
    res_poses = data_object["atom_positions"][:, CA_IDX, :]
    center_position = res_poses[is_mutation.bool()]
    _res_poses = torch.FloatTensor(res_poses)
    _center_position = torch.FloatTensor(center_position)
    center_indices = nearest(_res_poses, _center_position)

    try:
        dist_to_center = ((_res_poses - _center_position[center_indices])**2).sum(-1)
    except:
        raise ValueError(res_poses.shape, is_mutation, is_mutation.shape, center_indices, center_indices.shape, _res_poses.shape, _center_position.shape)
    dist_to_center[is_mutation.bool()] = 0.0
    selected_indices = torch.topk(dist_to_center, k, largest=False).indices
    selected_indices = selected_indices[selected_indices.argsort()]

    def consecutive_sequence(selected_index, half_gap_size = 1):
        new_idx = np.unique((selected_index + (np.arange(0, half_gap_size * 2 + 1) - half_gap_size).reshape(-1, 1)).reshape(-1))
        start = np.where(new_idx == selected_index.min())[0][0]
        end = np.where(new_idx == selected_index.max())[0][0] + 1
        new_idx = new_idx[start:end]
        return new_idx
                                                
    if consecutive:
        selected_indices = consecutive_sequence(selected_indices.numpy())

    data_object = get_subgraph(data_object, selected_indices)
    is_mutation = is_mutation[selected_indices]
    return data_object, is_mutation


def continues_crop(protein_feat, max_len):
    L = protein_feat['aatype'].shape[0]
    if L > max_len:
        start = np.random.randint(0, L - max_len + 1)
        end = start + max_len
        protein_feat = tree.map_structure(
                lambda x: x[start : end], protein_feat)
    return protein_feat


def spatial_crop(protein_feat, ligand_feat, max_len, interface: bool = False):
    def L2_dist(A: np.ndarray, B: np.ndarray):
        # return distance: [A.shape[0], B.shape[0]]
        assert len(A.shape) == 2 and len(B.shape) == 2
        diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))

        return distances

    if interface:
        if ligand_feat is not None:
            # case 1: prot-ligand
            prot_centre_coord = protein_feat['atom_positions'][:, CA_IDX, :]
            ligand_centre_coord = ligand_feat['ligand_atom_positions']
            if max_len >= prot_centre_coord.shape[0] + ligand_centre_coord.shape[0]:
                return protein_feat, ligand_feat
            
            distances = L2_dist(prot_centre_coord, ligand_centre_coord)
            prot_idx, ligand_idx = np.where(distances < 15)
            assert prot_idx.shape[0] > 0 and ligand_idx.shape[0] > 0
            prot_idx, ligand_idx = np.unique(prot_idx), np.unique(ligand_idx)
            random_select_idx = np.random.randint(
                0, prot_idx.shape[0] + ligand_idx.shape[0]
            )
            if random_select_idx < prot_idx.shape[0]:
                centre_coord = prot_centre_coord[prot_idx[random_select_idx], :]
            else:
                centre_coord = ligand_centre_coord[
                    ligand_idx[random_select_idx - prot_idx.shape[0]], :
                ]
            
            distances = L2_dist(centre_coord[np.newaxis, :], np.vstack([prot_centre_coord, ligand_centre_coord]))
            
            indices = np.argpartition(distances[0, :], max_len)[:max_len]
            prot_selected_indices = indices[np.where(indices < prot_centre_coord.shape[0])[0]]
            ligand_selected_indices = indices[
                np.where(indices >= prot_centre_coord.shape[0])[0]
            ] - prot_centre_coord.shape[0]
            prot_selected_indices = np.sort(prot_selected_indices) # important !!!
            ligand_selected_indices = np.sort(ligand_selected_indices) # important !!!

            protein_feat = get_subgraph(protein_feat, prot_selected_indices)
            ligand_feat = get_subgraph(ligand_feat, ligand_selected_indices)
        
        else:
            # case 2:prot-prot
            if protein_feat['aatype'].shape[0] <= max_len:
                return protein_feat, ligand_feat
            
            unique_chain_numbers = np.unique(protein_feat['chain_index'])
            assert unique_chain_numbers.shape[0] == 2
            chain_0_mask = np.where(protein_feat['chain_index'] == unique_chain_numbers[0])[0]
            chain_1_mask = np.where(protein_feat['chain_index'] == unique_chain_numbers[1])[0]

            prot_centre_coord_chain_0 = protein_feat['atom_positions'][chain_0_mask, CA_IDX, :]
            prot_centre_coord_chain_1 = protein_feat['atom_positions'][chain_1_mask, CA_IDX, :]
            prot_centre_coord = protein_feat['atom_positions'][:, CA_IDX, :]

            distances = L2_dist(prot_centre_coord_chain_0, prot_centre_coord_chain_1)
            prot_chain_0_idx, prot_chain_1_idx = np.where(distances < 15)
            assert prot_chain_0_idx.shape[0] > 0 and prot_chain_1_idx.shape[0] > 0

            prot_chain_0_idx, prot_chain_1_idx = np.unique(prot_chain_0_idx), np.unique(prot_chain_1_idx)
            random_select_idx = np.random.randint(
                0, prot_chain_0_idx.shape[0] + prot_chain_1_idx.shape[0]
            )
            if random_select_idx < prot_chain_0_idx.shape[0]:
                centre_coord = prot_centre_coord_chain_0[prot_chain_0_idx[random_select_idx], :]
            else:
                centre_coord = prot_centre_coord_chain_1[
                    prot_chain_1_idx[random_select_idx - prot_chain_0_idx.shape[0]], :
                ]
            
            distances = L2_dist(centre_coord[np.newaxis, :], prot_centre_coord)
            indices = np.argpartition(distances[0, :], max_len)[:max_len]
            indices = np.sort(indices) # important !!!
            protein_feat = get_subgraph(protein_feat, indices)
            
    else:
        assert ligand_feat is None, 'only single chain prot uses spatial crop (no interface)'
        if protein_feat['aatype'].shape[0] <= max_len:
            return protein_feat, ligand_feat
        
        prot_centre_coord = protein_feat['atom_positions'][:, CA_IDX, :]
        centre_idx = np.random.randint(0, prot_centre_coord.shape[0])
        A = prot_centre_coord[centre_idx][np.newaxis, :]
        distances = L2_dist(A, prot_centre_coord)[0, :]
        indices = np.argpartition(distances, max_len)[:max_len]
        indices = np.sort(indices) # important !!!
        protein_feat = get_subgraph(protein_feat, indices)

    return protein_feat, ligand_feat
