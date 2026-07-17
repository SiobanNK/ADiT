"""Length-based batch sampler for ADiT.

`BatchTensorConverter` (in `protein_datamodule.py`) collates a batch by padding
every tensor to the shape of the *longest* sample in that batch (dense pad,
no bucketing). With a fixed `batch_size`, this means:

  - batches full of small monomers waste GPU memory (over-padded relative to
    what they need), while
  - a batch that happens to contain one large PPI/antibody-antigen complex
    can OOM, since the whole batch gets padded up to it.

`DynamicBatchSampler` fixes this by grouping dataset indices into batches such
that `batch_size_in_batch * max_L_in_batch <= max_tokens_per_batch`, i.e. it
directly models the padded-batch cost instead of just capping the number of
samples. This is the same idea as fairseq's "max tokens" batching.

IMPORTANT -- length metric: `atom_positions`/`atom_mask` have shape (L, 37, ...),
where L = number of tokens (1 per protein residue + 1 per ligand atom, see
`FeatureTransform.merge_feats`). The 37 is a *fixed* per-token dimension (atom
slots within a residue/ligand-atom token), so it's L -- not the real atom count
(`atom_mask.sum()`) -- that sets the padded tensor's shape and therefore the
actual memory cost. The default length metric here is L (`seq_mask.shape[0]`,
or the sum of `seq_mask.shape[0]` across every `*_seq_mask` key, to handle
SKEMPI/HER2 samples that bundle a wt_ and mt_ structure together).

Usage is two-step:
    1. `compute_lengths(dataset)` (once, cache to disk) to get L per index.
    2. `DynamicBatchSampler(lengths, max_tokens_per_batch=...)` as the
       `batch_sampler=` of a `DataLoader`.
"""
from typing import Optional, Sequence, Callable

import os
import numpy as np
import torch
from torch.utils.data import DataLoader


def _default_length_fn(item: dict, length_key: str = "seq_mask") -> int:
    """L for one dataset item: shape[0] of `length_key`, summed over every
    key that is `length_key` or ends with `f"_{length_key}"`.

    Plain datasets (pretrain/LBA/Davis) only have `"seq_mask"` -> L of the
    single (protein [+ ligand]) sample. SKEMPI/HER2 have `"wt_seq_mask"` and
    `"mt_seq_mask"` -> L is the sum of both structures' lengths, since both
    get padded into their own dense tensors by `BatchTensorConverter` but
    both live in the same batch (see the previous discussion of this file).
    """
    total, found = 0, False
    suffix = "_" + length_key
    for k, v in item.items():
        if k == length_key or k.endswith(suffix):
            found = True
            if not torch.is_tensor(v):
                v = torch.as_tensor(v)
            total += int(v.shape[0])
    if not found:
        raise KeyError(
            f"No key named '{length_key}' or ending in '{suffix}' found in "
            f"item (keys: {list(item.keys())}). Pass a custom `length_fn` "
            f"or `length_key` matching your featurization."
        )
    return total


def compute_lengths(
    dataset,
    length_fn: Optional[Callable] = None,
    length_key: str = "seq_mask",
    num_workers: int = 0,
) -> np.ndarray:
    """Iterate once over `dataset` and record L (token count) per index.

    Pass `length_fn(item) -> int` instead of the default if you want a
    different notion of length (e.g. real atom count via
    `item["atom_mask"].sum()`, if you ever need to budget on that instead
    of on padded-tensor size).

    This calls `dataset[i]` for every i, so for datasets like `AditPDBDataset`
    where `__getitem__` re-samples a random cluster member + random crop, the
    returned lengths are only a proxy for what will actually be drawn during
    training -- still useful for bucketing since crops are capped by
    `truncate_length` and don't vary wildly. For deterministic datasets
    (LBA, Davis, SKEMPI, HER2), lengths are exact.

    Uses a plain `DataLoader` with `num_workers` so precomputation can be
    parallelized across CPU workers -- worth doing once and caching, see
    `scripts/precompute_lengths.py`, especially before a multi-node Jean-Zay
    job (don't let every rank recompute this in parallel on a shared
    filesystem).
    """
    if length_fn is None:
        def length_fn(item):
            return _default_length_fn(item, length_key=length_key)

    def _collate(batch):
        return [length_fn(item) for item in batch]

    loader = DataLoader(
        dataset, batch_size=1, num_workers=num_workers, collate_fn=_collate
    )
    lengths = []
    for batch_lengths in loader:
        lengths.extend(batch_lengths)
    return np.asarray(lengths, dtype=np.int64)


def compute_or_load_lengths(
    dataset,
    cache_path: Optional[str] = None,
    length_fn: Optional[Callable] = None,
    length_key: str = "seq_mask",
    num_workers: int = 0,
) -> np.ndarray:
    """`compute_lengths`, cached to a `.npy` file at `cache_path`.

    On Jean-Zay, prefer building this cache once with
    `scripts/precompute_lengths.py` (single task, e.g. in an interactive
    session or a 1-GPU job) *before* launching the actual multi-GPU/multi-node
    training job, then just point every rank at the same `cache_path` so
    they all read instead of racing to write.
    """
    if cache_path is not None and os.path.exists(cache_path):
        return np.load(cache_path)

    lengths = compute_lengths(
        dataset,
        length_fn=length_fn,
        length_key=length_key,
        num_workers=num_workers,
    )

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(cache_path, lengths)

    return lengths


class DynamicBatchSampler(torch.utils.data.Sampler):
    """Yields lists of dataset indices whose total attention cost stays under
    a fixed budget.

    ADiT's token-level attention (`SimplePairFormer` + `AttentionPairBias`,
    see `pairformer.py`/`diffusion_transformer.py`) builds its pair
    representation `z_ij` from *dense, per-sample* edges
    (`generate_dense_attention_edge_batch`), concatenated block-diagonally
    across the batch -- i.e. there is no edge (and no computation) between
    tokens of two different samples. So the total edge/pair-tensor cost of a
    batch is `sum(L_i ** 2)`, not `batch_size * max(L_i) ** 2`: a big complex
    does not "contaminate" the cost of the other samples sharing its batch,
    it only adds its own L^2.

    (The atom-level attention is windowed/local -- `generate_sparse_attention_edge_batch`
    -- so it's linear in atom count and not the binding constraint; the
    quadratic token-level pair representation dominates memory for anything
    but tiny samples.)

    Packing: samples are length-sorted (after shuffling, so ties don't always
    land in the same batch) and greedily packed while
    `sum(L_i ** 2 for i in batch) <= max_pair_budget`.

    Pass `cost_model="padded"` instead if you need the old, more conservative
    model (`batch_size * max_len <= max_tokens_per_batch`) -- e.g. if you
    later add a module that pads to `(B, L_max, ...)` rather than using
    block-diagonal edges.

    DDP: batches are built identically on every rank (same seed + epoch),
    then simply strided by `rank::num_replicas`, so ranks see disjoint
    batches without needing to communicate.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        max_pair_budget: Optional[int] = None,
        max_tokens_per_batch: Optional[int] = None,
        cost_model: str = "sum_of_squares",
        shuffle: bool = True,
        drop_last: bool = False,
        max_batch_size: Optional[int] = None,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
    ):
        assert 0 <= rank < num_replicas
        assert cost_model in ("sum_of_squares", "padded")
        if cost_model == "sum_of_squares":
            assert max_pair_budget is not None, (
                "cost_model='sum_of_squares' needs max_pair_budget "
                "(budget on sum(L_i**2) per batch)."
            )
        else:
            assert max_tokens_per_batch is not None, (
                "cost_model='padded' needs max_tokens_per_batch "
                "(budget on batch_size * max(L_i) per batch)."
            )
        self.lengths = np.asarray(lengths)
        self.max_pair_budget = max_pair_budget
        self.max_tokens_per_batch = max_tokens_per_batch
        self.cost_model = cost_model
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.max_batch_size = max_batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self._build_batches()

    def set_epoch(self, epoch: int) -> None:
        """Call before each epoch to reshuffle (mirrors DistributedSampler).

        With plain PyTorch Lightning you generally don't need to call this
        yourself: `ProteinDataModule` bumps its own epoch counter and rebuilds
        the sampler every time `train_dataloader()` is invoked, as long as you
        set `trainer.reload_dataloaders_every_n_epochs=1`.
        """
        self.epoch = epoch
        self._build_batches()

    def _build_batches(self) -> None:
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = np.arange(len(self.lengths))
        if self.shuffle:
            rng.shuffle(indices)
        order = indices[np.argsort(self.lengths[indices], kind="stable")]

        batches, cur_batch = [], []
        cur_cost, cur_max_len = 0, 0  # cur_cost: sum(L^2) so far; cur_max_len: for "padded" model
        for idx in order:
            length = int(self.lengths[idx])
            new_size = len(cur_batch) + 1

            if self.cost_model == "sum_of_squares":
                new_cost = cur_cost + length ** 2
                over_budget = new_cost > self.max_pair_budget
            else:
                new_max_len = max(cur_max_len, length)
                new_cost = new_size * new_max_len
                over_budget = new_cost > self.max_tokens_per_batch

            over_size_cap = (
                self.max_batch_size is not None and new_size > self.max_batch_size
            )

            if cur_batch and (over_budget or over_size_cap):
                batches.append(cur_batch)
                cur_batch = []
                cur_cost, cur_max_len = 0, 0

            cur_batch.append(int(idx))
            cur_cost += length ** 2
            cur_max_len = max(cur_max_len, length)

        if cur_batch and (not self.drop_last or len(batches) == 0):
            batches.append(cur_batch)

        if self.shuffle:
            perm = rng.permutation(len(batches))
            batches = [batches[i] for i in perm]

        self.batches = batches
        self.rank_batches = batches[self.rank :: self.num_replicas]

    def __iter__(self):
        return iter(self.rank_batches)

    def __len__(self) -> int:
        return len(self.rank_batches)
