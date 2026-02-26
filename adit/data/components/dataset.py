import os
import pickle
import csv
import tree
from pathlib import Path
from glob import glob
from typing import Optional, Sequence, List, Union
from functools import lru_cache

import numpy as np
import torch
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from adit.common import residue_constants
from adit.common.ligand import element_dict
from adit.common import residue_constants, data_transforms, rigid_utils, protein
from adit.data.components.feature_transform import FeatureTransform
from adit.data.components.feature_transform import FeatureTransform as ProteinFeatureTransform # to load old ckpt
from adit.data.components.metadata_filter import MetadataFilter
from adit.data.components.data_utils import read_csv, csv2dicts, extract_dest_protein_and_mutation_mask, truncation_skempi


class RandomAccessProteinDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 path_to_dataset: Union[Path, str],
                 path_to_esm_repr: str = None,
                 metadata_filter: Optional[MetadataFilter] = None,
                 transform: Optional[FeatureTransform] = None, 
    ):
        super().__init__()
        self.path_to_dataset = os.path.expanduser(path_to_dataset)
        if path_to_esm_repr:
            self.path_to_esm_repr = os.path.expanduser(path_to_esm_repr)
        else:
            self.path_to_esm_repr = None
        self.metadata_filter = metadata_filter
        self.transform = transform
    
    def split(self, train_val_ratio = None, generator_seed = None):
        assert hasattr(self, 'data') and hasattr(self, 'training'), "invoke 'set_training_mode' or 'set_testing_mode' first"
        assert self.training, "invoke 'set_training_mode' first"
        if self.train_indices is not None and self.val_indices is not None:
            train_set = torch.utils.data.Subset(self, self.train_indices)
            val_set = torch.utils.data.Subset(self, self.val_indices)
            return train_set, val_set
        else:
            assert train_val_ratio is not None and generator_seed is not None
            train_set, val_set = random_split(
                dataset=self,
                lengths=train_val_ratio,
                generator=torch.Generator().manual_seed(generator_seed),
            )
            return train_set, val_set
    
    def set_training_mode(self):
        # create self.data, self.train_indices, self.val_indices, self.training
        self.data = np.asarray([])
        self.train_indices, self.val_indices = None, None
        self.training = True
        raise NotImplementedError("Shoud be implemented in corresponding datatset object")

    def set_testing_mode(self):
        # create self.data, self.training
        self.data = np.asarray([])
        self.training = False
        raise NotImplementedError("Shoud be implemented in corresponding datatset object")

    @property    
    def num_samples(self):
        assert hasattr(self, 'data'), "invoke 'set_training_mode' or 'set_testing_mode' first"
        return len(self.data)
    
    def len(self): 
        return self.__len__()

    def __len__(self):
        return self.num_samples

    def get(self, idx):
        return self.__getitem__(idx)

    @lru_cache(maxsize=100)
    def __getitem__(self, idx):
        # base version, used by PDBDataset, LBADataset
        data_path = self.data[idx]
        accession_code = os.path.splitext(os.path.basename(data_path))[0]
        
        with open(data_path, 'rb') as f:
            data_object = pickle.load(f)

        if self.path_to_esm_repr:
            esm_repr_file = os.path.join(self.path_to_esm_repr, f"{accession_code}.pt")
            data_object["esm_repr"] = torch.load(esm_repr_file)["representations"][33]
        
        # Apply data transform
        if self.transform is not None:
            if not self.training:
                data_object = self.transform(data_object, val_stage = True)
            elif self.val_indices is not None and idx in self.val_indices:
                data_object = self.transform(data_object, val_stage = True)
            else:
                data_object = self.transform(data_object)
        
        data_object['accession_code'] =  accession_code
        return data_object  # dict of arrays


class AditPDBDataset(RandomAccessProteinDataset):

    def set_training_mode(self):
        # For pdb data.
        csv_path = os.path.join(self.path_to_dataset, 'processed_indices.csv')
        self._df = pd.read_csv(csv_path, low_memory=False)
        if self.metadata_filter:
            self._df = self.metadata_filter(self._df)
        
        self.data = [
            df_one_cluster.reset_index() for _, df_one_cluster in self._df.groupby("cluster_id", sort=True)
        ]
        self.train_indices, self.val_indices = None, None
        self.training = True
    
    def set_testing_mode(self):
        self.set_training_mode()
        self.training = False

    @lru_cache(maxsize=100)
    def __getitem__(self, idx):
        df_one_group = self.data[idx]
        df_one_group_samples = df_one_group.sample(n = 10, replace = True)
        for i in range(10):
            data_info_dict = df_one_group_samples.iloc[i].to_dict()
            crop_methods = [data_info_dict['crop_method_1'], data_info_dict['crop_method_2']]
            crop_methods = [m for m in crop_methods if m == m]
            ref_chains = [
                int(data_info_dict['prot_chain_1']), 
                int(data_info_dict['prot_chain_2']), 
                int(data_info_dict['ligand_chain'])
            ]
            accession_code = '.'.join([
                data_info_dict['pdb_id'],
                str(data_info_dict['prot_chain_1']),
                str(data_info_dict['prot_chain_2']),
                str(data_info_dict['ligand_chain'])
            ])
            data_path = os.path.join(
                self.path_to_dataset, 'adit', data_info_dict['pdb_id'] + '.pkl'
            )
            
            with open(data_path, 'rb') as f:
                data_object = pickle.load(f)
            
            try:
                # Apply data transform
                if self.transform is not None:
                    data_object = self.transform(data_object, ref_chains, crop_methods)
            except:
                continue
            
            data_object['accession_code'] =  accession_code
            return data_object  # dict of arrays
        
        return self.__getitem__(np.random.randint(0, len(self)))


class LBADataset(RandomAccessProteinDataset):

    def __init__(self, 
                 path_to_dataset: Union[Path, str],
                 transform: Optional[FeatureTransform] = None, 
                 split_identity_threshold: str = 'identity_30'
    ):
        super().__init__(path_to_dataset, None, None, transform)
        self.split_dir = os.path.join(
            path_to_dataset, f"lba_{split_identity_threshold}_indices"
        )
        self.data_dir = os.path.join(path_to_dataset, "processed_LBA")
        
    def set_training_mode(self):
        with open(os.path.join(self.split_dir, "train_indices.txt"), "r") as f:
            training_ids = f.readlines()
        training_ids = [id.strip() for id in training_ids]
        self.training_data = [
            os.path.join(self.data_dir, f"{id}.pkl")
            for id in training_ids
        ]

        with open(os.path.join(self.split_dir, "val_indices.txt"), "r") as f:
            val_ids = f.readlines()
        val_ids = [id.strip() for id in val_ids]
        self.validation_data = [
            os.path.join(self.data_dir, f"{id}.pkl")
            for id in val_ids
        ]
                
        self.train_indices = list(range(len(self.training_data)))
        self.val_indices = list(range(len(self.training_data), len(self.training_data) + len(self.validation_data)))
        self.data = np.asarray(self.training_data + self.validation_data)
        self.training = True

    def set_testing_mode(self):
        with open(os.path.join(self.split_dir, "test_indices.txt"), "r") as f:
            test_ids = f.readlines()
        test_ids = [id.strip() for id in test_ids]
        self.test_data = [
            os.path.join(self.data_dir, f"{id}.pkl")
            for id in test_ids
        ]
        self.data = np.asarray(self.test_data)
        self.training = False


class DavisDataset(RandomAccessProteinDataset):

    def __init__(self, 
                 path_to_dataset: Union[Path, str],
                 transform: Optional[FeatureTransform] = None, 
                 split_seed = "seed_1"
    ):
        super().__init__(path_to_dataset, None, None, transform)
        self.split_dir = os.path.join(
            path_to_dataset, f"train_val_random_{split_seed}"
        )
        self.protein_dir = os.path.join(
            path_to_dataset, "processed_davis_proteins"
        )
        self.simles_dir = os.path.join(
            path_to_dataset, "smiles"
        )

    def set_training_mode(self):
        self.training_data = read_csv(os.path.join(self.split_dir, "df_train.csv"))
        self.validation_data = read_csv(os.path.join(self.split_dir, "df_valid.csv"))
                
        self.train_indices = list(range(len(self.training_data)))
        self.val_indices = list(range(len(self.training_data), len(self.training_data) + len(self.validation_data)))

        self.data = np.asarray(self.training_data + self.validation_data)
        self.training = True

    def set_testing_mode(self):
        self.test_data = read_csv(os.path.join(self.split_dir, "df_test.csv"))
        self.data = np.asarray(self.test_data)
        self.training = False

    @lru_cache(maxsize=100)
    def __getitem__(self, idx):
        data_info = self.data[idx]
        
        smiles_path = os.path.join(self.simles_dir, data_info[-2] + ".pkl")
        protein_path = os.path.join(self.protein_dir, data_info[-1] + ".pkl")
        
        with open(smiles_path, 'rb') as f:
            ligand_data_object = pickle.load(f)
        
        with open(protein_path, 'rb') as f:
            data_object = pickle.load(f)

        if self.transform is not None:
            data_object = self.transform((data_object, ligand_data_object, float(data_info[-3])))
        
        return data_object  # dict of arrays


class SkempiDataset(RandomAccessProteinDataset):

    def __init__(self, 
                 path_to_dataset: Union[Path, str],
                 path_to_esm_repr: Union[Path, str],
                 transform: Optional[FeatureTransform] = None, 
                 test_split: str = "split_0",
                 skempi_truncate: bool = True,
                 truncation_size: int = 100,
                 consecutive: bool = False
    ):
        super().__init__(path_to_dataset, path_to_esm_repr, None, transform)
        self.test_split = test_split
        self.splits = glob(os.path.join(path_to_dataset, "*"))
        self.splits = {
            os.path.basename(one_split): glob(os.path.join(one_split, '*.pkl'))
            for one_split in self.splits
        }
        self.skempi_truncate = skempi_truncate
        self.truncation_size = truncation_size
        self.consecutive = consecutive
    
    def set_training_mode(self):
        self.training_data = []
        self.validation_data = []
        for k in self.splits.keys():
            if k != self.test_split:
                self.training_data = self.training_data + self.splits[k]
            else:
                self.validation_data = self.validation_data + self.splits[k]
                
        self.train_indices = list(range(len(self.training_data)))
        self.val_indices = list(range(len(self.training_data)))
        self.data = np.asarray(self.training_data + self.validation_data)
        self.training = True

    def set_testing_mode(self):
        self._data = self.splits[self.test_split]
        self.data = np.asarray(self._data)
        self.training = False

    def PPI_base_info(self, idx):
        data_info = self.data[idx]
        with open(data_info, 'rb') as f:
            entry = pickle.load(f)
        wt_protein = entry["wt_data_object"]
        mt_protein = entry["mt_data_object"]
        ddG = entry["ddG"]
        accession_code = os.path.basename(data_info)[:-4]
        mutations = entry["mutation"].split(",")
        chain_a = entry["chain_a"]
        chain_b = entry["chain_b"]

        if self.path_to_esm_repr:
            wt_esm_repr_file = os.path.join(self.path_to_esm_repr, "wt_%s.pt" % accession_code)
            wt_protein["esm_repr"] = torch.load(wt_esm_repr_file)

            mt_esm_repr_file = os.path.join(self.path_to_esm_repr, "mt_%s.pt" % accession_code)
            mt_protein["esm_repr"] = torch.load(mt_esm_repr_file)

        return wt_protein, mt_protein, ddG, accession_code, mutations, chain_a, chain_b

    @lru_cache(maxsize=100)
    def __getitem__(self, idx):      
        wt_protein, mt_protein, ddG, accession_code, mutations, chain_a, chain_b = self.PPI_base_info(idx)

        if self.skempi_truncate:
            wt_protein, is_mutation_wild_type = extract_dest_protein_and_mutation_mask(
                wt_protein, chain_a, chain_b, mutations
            )
            mt_protein, is_mutation_mutant = extract_dest_protein_and_mutation_mask(
                mt_protein, chain_a, chain_b, mutations
            )
            wt_protein, is_mutation_wild_type = truncation_skempi(
                wt_protein, is_mutation_wild_type, self.truncation_size, self.consecutive
            )
            mt_protein, is_mutation_mutant = truncation_skempi(
                mt_protein, is_mutation_mutant, self.truncation_size, self.consecutive
            )
        else:
            del wt_protein["original_index"]
            del wt_protein["chain_id_mapping"]
            del mt_protein["original_index"]
            del mt_protein["chain_id_mapping"]
            
        # Apply data transform
        if self.transform is not None:
            wt_protein = self.transform(wt_protein)
            mt_protein = self.transform(mt_protein)
        
        def add_prefix(_d: dict, prefix: str):
            d = {
                prefix + "_" + k: _d[k]
                for k in _d.keys()
            }
            return d
        wt_protein = add_prefix(wt_protein, "wt")
        mt_protein = add_prefix(mt_protein, "mt")
        
        data_object = {}
        data_object.update(wt_protein)
        data_object.update(mt_protein)
        data_object["ddG"] = torch.FloatTensor([ddG])
        data_object["accession_code"] = accession_code
        
        return data_object  # dict of arrays


class HER2Dataset(SkempiDataset):
    processed_data_file = "processed_HER2_binders.csv"
    has_target_ddG = True

    def __init__(self, 
                 path_to_dataset: Union[Path, str],
                 path_to_esm_repr: Union[Path, str],
                 transform: Optional[FeatureTransform] = None, 
                 skempi_truncate: bool = True,
                 truncation_size: int = 50,
                 consecutive: bool = False
    ):
        super(SkempiDataset, self).__init__(
            path_to_dataset, path_to_esm_repr, None, transform
        ) # skip SkempiDataset.__init__()

        self.skempi_truncate = skempi_truncate
        self.truncation_size = truncation_size
        self.consecutive = consecutive
    
    def set_training_mode(self):
        raise ValueError("HER2/Antibody only for test")
    
    def set_testing_mode(self):
        self._data = csv2dicts(
            os.path.join(
                self.path_to_dataset, self.processed_data_file
            )
        )
        self.data = np.asarray(self._data)
        self.training = False
    
    def PPI_base_info(self, idx):
        data_info = self.data[idx]
        mt_protein_path = os.path.join(
            self.path_to_dataset, "processed", data_info["mt_protein"][:-3] + "pkl"
        )
        wt_protein_path = os.path.join(
            self.path_to_dataset, "processed", data_info["wt_protein"][:-3] + "pkl"
        )
        
        with open(mt_protein_path, 'rb') as f:
            mt_protein = pickle.load(f)
        with open(wt_protein_path, 'rb') as f:
            wt_protein = pickle.load(f)
        if self.has_target_ddG:
            ddG = float(data_info["ddG"])
        else:
            ddG = 0.

        if self.path_to_esm_repr:
            wt_esm_repr_file = os.path.join(self.path_to_esm_repr, "%s.pt" % os.path.splitext(data_info["wt_protein"])[0])
            wt_protein["esm_repr"] = torch.load(wt_esm_repr_file)

            mt_esm_repr_file = os.path.join(self.path_to_esm_repr, "%s.pt" % os.path.splitext(data_info["mt_protein"])[0])
            mt_protein["esm_repr"] = torch.load(mt_esm_repr_file)
        
        accession_code = os.path.splitext(data_info["wt_protein"])[0] + '_' + os.path.splitext(data_info["mt_protein"])[0]
        mutations = data_info["mutation"].split(",")
        chain_a = data_info["chain_a"]
        chain_b = data_info["chain_b"]

        return wt_protein, mt_protein, ddG, accession_code, mutations, chain_a, chain_b


class AntibodyDataset(HER2Dataset):
    processed_data_file = "processed_data.csv"
    has_target_ddG = False
    
    def PPI_base_info(self, idx):
        data_info = self.data[idx]
        mt_protein_path = os.path.join(
            self.path_to_dataset, "processed", data_info["mt_protein"][:-3] + "pkl"
        )
        wt_protein_path = os.path.join(
            self.path_to_dataset, "processed", data_info["wt_protein"][:-3] + "pkl"
        )
        
        with open(mt_protein_path, 'rb') as f:
            mt_protein = pickle.load(f)
        with open(wt_protein_path, 'rb') as f:
            wt_protein = pickle.load(f)
        if self.has_target_ddG:
            ddG = float(data_info["ddG"])
        else:
            ddG = 0.
        if self.path_to_esm_repr:
            wt_esm_repr_file = os.path.join(self.path_to_esm_repr, "%s.pt" % os.path.splitext(data_info["wt_protein"])[0])
            wt_protein["esm_repr"] = torch.load(wt_esm_repr_file)

            mt_esm_repr_file = os.path.join(self.path_to_esm_repr, "%s.pt" % os.path.splitext(data_info["mt_protein"])[0])
            mt_protein["esm_repr"] = torch.load(mt_esm_repr_file)
            
        accession_code = data_info["mt_protein"][:-3] + '_' + data_info["wt_protein"][:-3] +  data_info["mutation"]
        mutations = data_info["mutation"].split(",")
        chain_a = data_info["chain_a"]
        chain_b = data_info["chain_b"]

        return wt_protein, mt_protein, ddG, accession_code, mutations, chain_a, chain_b
