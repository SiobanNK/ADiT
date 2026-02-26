from typing import Optional, List
import pandas as pd
import numpy as np


class MetadataFilter:
    def __init__(self, 
                 min_len: Optional[int] = None,
                 max_len: Optional[int] = None,
                 min_chains: Optional[int] = None,
                 max_chains: Optional[int] = None,
                 min_resolution: Optional[int] = None,
                 max_resolution: Optional[int] = None,
                 include_structure_method: Optional[List[str]] = None,
                 include_oligomeric_detail: Optional[List[str]] = None,
                 **kwargs,
    ):
        self.min_len = min_len
        self.max_len = max_len
        self.min_chains = min_chains
        self.max_chains = max_chains
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.include_structure_method = include_structure_method
        self.include_oligomeric_detail = include_oligomeric_detail
    
    def __call__(self, df):
        _pre_filter_len = len(df)
        if self.min_len is not None:
            df = df[df['raw_seq_len'] >= self.min_len]
        if self.max_len is not None:
            df = df[df['raw_seq_len'] <= self.max_len]
        if self.min_chains is not None:
            df = df[df['num_chains'] >= self.min_chains]
        if self.max_chains is not None:
            df = df[df['num_chains'] <= self.max_chains]
        if self.min_resolution is not None:
            df = df[df['resolution'] >= self.min_resolution]
        if self.max_resolution is not None:
            df = df[df['resolution'] <= self.max_resolution]
        if self.include_structure_method is not None:
            df = df[df['include_structure_method'].isin(self.include_structure_method)]
        if self.include_oligomeric_detail is not None:
            df = df[df['include_oligomeric_detail'].isin(self.include_oligomeric_detail)]
        
        print(f">>> Filter out {len(df)} samples out of {_pre_filter_len} by the metadata filter")
        return df


class AditMetadataFilter(MetadataFilter):
    
    def __call__(self, df):
        df = df.copy()
        df = df.dropna(subset=['prot_chain_1_length', 'prot_chain_2_length', 'ligand_chain_length'])

        columns_to_sum = ['prot_chain_1_length', 'prot_chain_2_length']
        for col in columns_to_sum:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
        df['sum'] = df[columns_to_sum].sum(axis=1)

        if self.min_len is not None:
            df = df[df['sum'] >= self.min_len].drop(columns=['sum'])

        return df