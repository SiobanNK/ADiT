from typing import Any, Dict, Optional, Tuple, List, Sequence

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from lightning import LightningDataModule
from hydra.utils import instantiate

from adit.data.components.samplers import DynamicBatchSampler, compute_or_load_lengths


class BatchTensorConverter:
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """
    def __init__(self, target_keys: Optional[List] = None):
        self.target_keys = target_keys

    def __call__(self, raw_batch: Sequence[Dict[str, object]]):
        B = len(raw_batch)
        # Only do for Tensor
        target_keys = self.target_keys \
            if self.target_keys is not None else [k for k,v in raw_batch[0].items() if torch.is_tensor(v)]
        # Non-array, for example string, int
        non_array_keys = [k for k in raw_batch[0] if k not in target_keys]
        collated_batch = dict()
        for k in target_keys:
            collated_batch[k] = self.collate_dense_tensors([d[k] for d in raw_batch], pad_v=0.0)
        for k in non_array_keys:    # return non-array keys as is
            collated_batch[k] = [d[k] for d in raw_batch]

        return collated_batch

    @staticmethod
    def collate_dense_tensors(samples: Sequence, pad_v: float = 0.0):
        """
        Takes a list of tensors with the following dimensions:
            [(d_11,       ...,           d_1K),
             (d_21,       ...,           d_2K),
             ...,
             (d_N1,       ...,           d_NK)]
        and stack + pads them into a single tensor of:
        (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
        """
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            result_i[tuple(slice(0, k) for k in t.shape)] = t
        return result


class ProteinDataModule(LightningDataModule):
    """`LightningDataModule` for a single protein dataset,
        for pretrain or finetune purpose.

    Two batching modes:

    - Fixed `batch_size` (default, unchanged behaviour): every batch has the
      same number of samples, whatever their length.
    - Budgeted (`max_pair_budget` set): every batch has ~the same total
      attention cost instead, via `DynamicBatchSampler`. Useful here because
      ADiT samples range from small monomers to multi-chain PPI/antibody-
      antigen complexes, so a fixed sample count either under-uses memory or
      OOMs depending on what lands in the batch. `batch_size` is then ignored
      (kept only as a fallback / sanity cap via `max_batch_size`).

      Default cost model is `sum_of_squares`: budgets on `sum(L_i**2)` over
      the batch, matching ADiT's token-level attention, which builds its pair
      representation from dense per-sample (block-diagonal) edges -- so a
      big complex adds its own L^2 to the batch cost, it does not multiply
      the cost of every other sample in the batch the way padding would.
      Pass `cost_model="padded"` + `max_tokens_per_batch` instead for the
      more conservative `batch_size * max_L` model.

      Set `max_total_atoms` too (recommended whenever `max_pair_budget` is
      set): ADiT's *atom-level* attention is windowed/local (fixed `N_query`/
      `N_key`), so its cost scales linearly with the batch's *total* atom
      count, not with any single sample's L. `max_pair_budget` alone doesn't
      bound this -- a batch of many small samples can pass the token-level
      budget while still holding far more atoms than the atom-level module
      can handle, OOMing there specifically. `max_total_atoms` adds a second,
      linear budget (`sum(atoms_i) <= max_total_atoms`) enforced jointly with
      `max_pair_budget`.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 64,
        generator_seed: int = 42,
        train_val_split: Tuple[float, float] = (0.95, 0.05),
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False,
        max_pair_budget: Optional[int] = None,
        max_tokens_per_batch: Optional[int] = None,
        cost_model: str = "sum_of_squares",
        max_total_atoms: Optional[int] = None,
        atom_key: str = "atom_mask",
        max_batch_size: Optional[int] = None,
        length_key: str = "seq_mask",
        length_cache_dir: Optional[str] = None,
    ) -> None:
        """Initialize a `ProteinDataModule`.

        :param batch_size: Samples per batch. Ignored if `max_pair_budget` or
            `max_tokens_per_batch` is set (kept only for the DDP-divisibility
            check / as a display value).
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param max_pair_budget: If set (with `cost_model="sum_of_squares"`,
            the default), switch to dynamic batching budgeted on
            `sum(L_i**2) <= max_pair_budget`, matching ADiT's token-level
            attention cost. See class docstring for why this (not a padded
            `n * max_L` model) is the right proxy here.
        :param max_tokens_per_batch: Budget for `cost_model="padded"`
            (`n_samples * max_L_in_batch <= max_tokens_per_batch`). Only used
            when `cost_model="padded"`.
        :param cost_model: `"sum_of_squares"` (default) or `"padded"`. See
            `DynamicBatchSampler` docstring.
        :param max_total_atoms: If set, also budget on
            `sum(atoms_i) <= max_total_atoms` over the batch, to bound the
            atom-level (windowed) attention module's memory -- see class
            docstring. Strongly recommended alongside `max_pair_budget`.
        :param atom_key: Key whose `.sum()` gives the real atom count for a
            sample (default `"atom_mask"`). Keys named `atom_key` or ending
            in `f"_{atom_key}"` are all summed (SKEMPI/HER2's `wt_`/`mt_`).
            Only used when `max_total_atoms` is set.
        :param max_batch_size: Optional safety cap on the number of samples in
            a dynamically-built batch (e.g. to avoid huge batches of tiny
            samples). Only used when dynamic batching is active.
        :param length_key: Key whose `.shape[0]` gives L for a sample
            (default `"seq_mask"`, matching ADiT's unified feature dict). Keys
            named `length_key` or ending in `f"_{length_key}"` are all summed
            (handles SKEMPI/HER2 samples, which bundle a `wt_seq_mask` and
            `mt_seq_mask`). Only used when dynamic batching is active.
        :param length_cache_dir: Directory to cache per-split length arrays
            (`train_lengths.npy`, `val_lengths.npy`, `test_lengths.npy`) so
            they aren't recomputed every run. On Jean-Zay, precompute this
            once (see `scripts/precompute_lengths.py`) before launching a
            multi-GPU/multi-node job so ranks don't race to write the cache.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset = dataset

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self._epoch = 0

        # print(f"Dataset size: {len(self.dataset)}")

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if stage == 'fit' and not self.data_train and not self.data_val:
            # dataset = ConcatDataset(datasets=[trainset, testset])
            if hasattr(self.dataset, "set_training_mode"):
                self.dataset.set_training_mode()
            if hasattr(self.dataset, "split"):
                self.data_train, self.data_val = self.dataset.split(self.hparams.train_val_split, self.hparams.generator_seed)
            else:
                self.data_train, self.data_val = random_split(
                    dataset=self.dataset,
                    lengths=self.hparams.train_val_split,
                    generator=torch.Generator().manual_seed(self.hparams.generator_seed),
                )
            print(f"Training set size: {len(self.data_train)}")
            print(f"Val set size: {len(self.data_val)}")
        elif stage in ('predict', 'test'):
            if hasattr(self.dataset, "set_testing_mode"):
                self.dataset.set_testing_mode()
            self.data_test = self.dataset
            print(f"Test set size: {len(self.data_test)}")
        else:
            raise NotImplementedError(f"Stage {stage} not implemented.")

    def _lengths_for(self, dataset: Dataset[Any], split: str):
        """Returns (lengths, atom_counts). `atom_counts` is None unless
        `max_total_atoms` is set.
        """
        cache_path = None
        if self.hparams.length_cache_dir is not None:
            suffix = "_with_atoms" if self.hparams.max_total_atoms is not None else ""
            cache_path = f"{self.hparams.length_cache_dir}/{split}_lengths{suffix}.npy"

        out = compute_or_load_lengths(
            dataset,
            cache_path=cache_path,
            length_key=self.hparams.length_key,
            atom_key=self.hparams.atom_key if self.hparams.max_total_atoms is not None else None,
            num_workers=self.hparams.num_workers,
        )
        if self.hparams.max_total_atoms is not None:
            return out[:, 0], out[:, 1]  # lengths, atom_counts
        return out, None

    def _dataloader_template(
        self, dataset: Dataset[Any], split: str, train: bool = True
    ) -> DataLoader[Any]:
        """Create a dataloader from a dataset.

        :param dataset: The dataset.
        :param split: "train" / "val" / "test", used for the length cache filename.
        :return: The dataloader.
        """
        batch_collator = BatchTensorConverter()    # list of dicts -> dict of tensors

        dynamic = (
            self.hparams.max_pair_budget is not None
            or self.hparams.max_tokens_per_batch is not None
        )
        if not dynamic:
            return DataLoader(
                dataset=dataset,
                collate_fn=batch_collator,
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=(self.hparams.shuffle and train),
            )

        world_size = self.trainer.world_size if self.trainer is not None else 1
        rank = self.trainer.global_rank if self.trainer is not None else 0

        lengths, atom_counts = self._lengths_for(dataset, split)
        batch_sampler = DynamicBatchSampler(
            lengths=lengths,
            max_pair_budget=self.hparams.max_pair_budget,
            max_tokens_per_batch=self.hparams.max_tokens_per_batch,
            cost_model=self.hparams.cost_model,
            atom_counts=atom_counts,
            max_total_atoms=self.hparams.max_total_atoms,
            shuffle=(self.hparams.shuffle and train),
            max_batch_size=self.hparams.max_batch_size,
            num_replicas=world_size,
            rank=rank,
            seed=self.hparams.generator_seed,
        )
        # NB: reshuffled every call, so set `trainer.reload_dataloaders_every_n_epochs=1`
        # in your Hydra config to actually get a new packing each epoch.
        batch_sampler.set_epoch(self._epoch)
        if train:
            self._epoch += 1

        return DataLoader(
            dataset=dataset,
            collate_fn=batch_collator,
            batch_sampler=batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return self._dataloader_template(self.data_train, split="train", train=True)


    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return self._dataloader_template(self.data_val, split="val", train=False)

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return self._dataloader_template(self.data_test, split="test", train=False)

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {"epoch": self._epoch}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        self._epoch = state_dict.get("epoch", 0)
