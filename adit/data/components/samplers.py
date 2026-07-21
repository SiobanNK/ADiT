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


def _reduce_matching_keys(item: dict, key: str, reduce: str) -> int:
    """Sum `reduce(v)` (`v.shape[0]` or `v.sum()`) over every key in `item`
    that is `key` or ends with `f"_{key}"` (handles SKEMPI/HER2's wt_/mt_
    prefixes, where both structures should count).
    """
    total, found = 0, False
    suffix = "_" + key
    for k, v in item.items():
        if k == key or k.endswith(suffix):
            found = True
            if not torch.is_tensor(v):
                v = torch.as_tensor(v)
            total += int(v.shape[0]) if reduce == "shape0" else int(v.sum().item())
    if not found:
        raise KeyError(
            f"No key named '{key}' or ending in '{suffix}' found in item "
            f"(keys: {list(item.keys())}). Pass a custom `length_fn`/`atom_count_fn`."
        )
    return total


def _default_length_fn(item: dict, length_key: str = "seq_mask") -> int:
    """L (token count) for one dataset item: sum of `shape[0]` over every
    key that is `length_key` or ends with `f"_{length_key}"`.

    Plain datasets (pretrain/LBA/Davis) only have `"seq_mask"` -> L of the
    single (protein [+ ligand]) sample. SKEMPI/HER2 have `"wt_seq_mask"` and
    `"mt_seq_mask"` -> L is the sum of both structures' lengths.
    """
    return _reduce_matching_keys(item, length_key, reduce="shape0")


def _default_atom_count_fn(item: dict, atom_key: str = "atom_mask") -> int:
    """Real atom count for one dataset item: sum of `.sum()` over every key
    that is `atom_key` or ends with `f"_{atom_key}"` (again handling SKEMPI/
    HER2's wt_/mt_ prefixes).
    """
    return _reduce_matching_keys(item, atom_key, reduce="sum")


def compute_lengths(
    dataset,
    length_fn: Optional[Callable] = None,
    length_key: str = "seq_mask",
    atom_count_fn: Optional[Callable] = None,
    atom_key: Optional[str] = None,
    num_workers: int = 0,
):
    """Iterate once over `dataset` and record, per index, L (token count) and
    optionally the real atom count.

    Pass `length_fn(item) -> int` / `atom_count_fn(item) -> int` instead of
    the defaults for a custom notion of either. Set `atom_key` (e.g.
    `"atom_mask"`) to also compute atom counts in the same pass -- needed for
    `DynamicBatchSampler`'s `max_total_atoms` budget (see its docstring for
    why the token-level `max_pair_budget` alone doesn't bound atom-level
    attention memory).

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

    Returns an (N,) array if `atom_key` is None, else an (N, 2) array with
    columns `[token_length, atom_count]`.
    """
    if length_fn is None:
        def length_fn(item):
            return _default_length_fn(item, length_key=length_key)
    if atom_key is not None and atom_count_fn is None:
        def atom_count_fn(item):
            return _default_atom_count_fn(item, atom_key=atom_key)

    def _collate(batch):
        if atom_key is None:
            return [length_fn(item) for item in batch]
        return [(length_fn(item), atom_count_fn(item)) for item in batch]

    loader = DataLoader(
        dataset, batch_size=1, num_workers=num_workers, collate_fn=_collate
    )
    out = []
    for batch_out in loader:
        out.extend(batch_out)
    if atom_key is None:
        return np.asarray(out, dtype=np.int64)
    return np.asarray(out, dtype=np.int64)  # shape (N, 2): [length, atom_count]


def compute_or_load_lengths(
    dataset,
    cache_path: Optional[str] = None,
    length_fn: Optional[Callable] = None,
    length_key: str = "seq_mask",
    atom_count_fn: Optional[Callable] = None,
    atom_key: Optional[str] = None,
    num_workers: int = 0,
):
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
        atom_count_fn=atom_count_fn,
        atom_key=atom_key,
        num_workers=num_workers,
    )

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(cache_path, lengths)

    return lengths


class DynamicBatchSampler(torch.utils.data.Sampler):
    """Yields lists of dataset indices whose total attention cost stays under
    fixed budget(s).

    ADiT's forward pass has two attention modules with different cost
    profiles, both of which need to be budgeted for packing to be safe:

    - Token-level (`SimplePairFormer` + the token `DiffusionTransformer`):
      dense, per-sample (block-diagonal) edges -- `generate_dense_attention_edge_batch`.
      A big complex adds its own `L**2` to the batch cost, it does not
      multiply the cost of every other sample in the batch the way padding
      would. Budgeted via `max_pair_budget`: `sum(L_i ** 2) <= max_pair_budget`.

    - Atom-level (the atom `DiffusionTransformer`, windowed/local attention
      via `generate_sparse_attention_edge_batch` with fixed `N_query`/`N_key`):
      each atom attends to at most `N_key` neighbours *regardless of protein
      size*, so its edge count -- and the resulting message tensors, e.g.
      `A_ij[..., None] * v_i[edge[1]]` in `AttentionPairBias` -- scale
      *linearly* with the **total atom count of the whole batch**, not with
      any single sample's length. A batch packed only against
      `max_pair_budget` can pass that check while still holding e.g. 90+
      small samples, whose atoms sum to something huge -> OOM in the
      atom-level module specifically (not the token-level one). Budgeted via
      `max_total_atoms`: `sum(atoms_i) <= max_total_atoms`.

    Set whichever budget(s) are relevant; a sample is only added to the
    current batch if *all* set budgets remain satisfied.

    Packing: samples are length-sorted (after shuffling, so ties don't always
    land in the same batch) and greedily packed while every active budget
    holds.

    Pass `cost_model="padded"` + `max_tokens_per_batch` instead of
    `max_pair_budget` if you need the old, more conservative
    `batch_size * max_L <= max_tokens_per_batch` model for the token side.

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
        atom_counts: Optional[Sequence[int]] = None,
        max_total_atoms: Optional[int] = None,
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
        if max_total_atoms is not None:
            assert atom_counts is not None, (
                "max_total_atoms needs atom_counts (per-index real atom "
                "count, e.g. from compute_lengths(..., atom_key='atom_mask'))."
            )
        self.lengths = np.asarray(lengths)
        self.atom_counts = np.asarray(atom_counts) if atom_counts is not None else None
        self.max_pair_budget = max_pair_budget
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_total_atoms = max_total_atoms
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
        cur_pair_cost, cur_max_len, cur_atom_total = 0, 0, 0
        for idx in order:
            length = int(self.lengths[idx])
            atoms = int(self.atom_counts[idx]) if self.atom_counts is not None else 0
            new_size = len(cur_batch) + 1

            if self.cost_model == "sum_of_squares":
                new_pair_cost = cur_pair_cost + length ** 2
                over_pair_budget = new_pair_cost > self.max_pair_budget
            else:
                new_max_len = max(cur_max_len, length)
                new_pair_cost = new_size * new_max_len
                over_pair_budget = new_pair_cost > self.max_tokens_per_batch

            over_atom_budget = (
                self.max_total_atoms is not None
                and (cur_atom_total + atoms) > self.max_total_atoms
            )
            over_size_cap = (
                self.max_batch_size is not None and new_size > self.max_batch_size
            )

            if cur_batch and (over_pair_budget or over_atom_budget or over_size_cap):
                batches.append(cur_batch)
                cur_batch = []
                cur_pair_cost, cur_max_len, cur_atom_total = 0, 0, 0

            cur_batch.append(int(idx))
            cur_pair_cost += length ** 2
            cur_max_len = max(cur_max_len, length)
            cur_atom_total += atoms

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
