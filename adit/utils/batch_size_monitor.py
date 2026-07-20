from typing import Any, Optional

import lightning as L
from lightning import Callback

from adit.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class BatchSizeMonitor(Callback):
    """Logs the number of atoms and tokens in every training batch, on every
    rank, to whatever logger(s) are configured (W&B, CSV, TensorBoard, ...).

    Unlike JobGPUMonitor (which polls on an interval from a background
    thread), this logs synchronously on each `on_train_batch_end`, since
    what explains a given step's peak memory is *that step's* batch size,
    not an average over the last N seconds.

    Logged per step (all under the `batch_size/` prefix):
      - batch_size/num_atoms          total atoms in the batch
      - batch_size/num_tokens         total tokens in the batch
      - batch_size/num_samples        number of molecules/structures/sequences
      - batch_size/max_atoms_per_sample   largest single sample (padding driver)
      - batch_size/max_tokens_per_sample  largest single sample (padding driver)

    NOTE: the extraction logic in `_extract_sizes` is written defensively
    for a few common batch layouts (PyG-style `Batch`, dict-of-tensors,
    plain objects with attributes) but ADiT's exact batch format wasn't
    available when writing this — check the TODOs in `_extract_sizes` and
    adjust the attribute/key names to match your actual collate output.
    """

    def __init__(self, log_every_n_steps: int = 1, prefix: str = "batch_size") -> None:
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.prefix = prefix

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx % self.log_every_n_steps != 0:
            return

        loggers = trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]
        loggers = [lg for lg in loggers if lg is not None]
        if not loggers:
            return

        sizes = self._extract_sizes(batch)
        if sizes is None:
            log.warning("BatchSizeMonitor: could not extract batch sizes, skipping.")
            return

        metrics = {f"{self.prefix}/{k}": v for k, v in sizes.items()}
        step = trainer.global_step
        for lg in loggers:
            try:
                lg.log_metrics(metrics, step=step)
            except Exception as e:
                log.warning(f"BatchSizeMonitor: failed to log to {type(lg).__name__}: {e}")

    @staticmethod
    def _extract_sizes(batch: Any) -> Optional[dict]:
        """Best-effort extraction of atom/token counts from a batch.

        Tries, in order:
          1. PyG-style `Batch` object: `.batch` is a per-atom graph-index
             tensor (len == total atoms, max index + 1 == num graphs);
             tokens fall back to atoms if no separate token field exists.
          2. dict-style batch with common key names.
          3. plain object with common attribute names.

        Returns a dict with keys: num_atoms, num_tokens, num_samples,
        max_atoms_per_sample, max_tokens_per_sample — or None if nothing matched.
        """
        try:
            # --- 1. PyG-style geometric batch -----------------------------
            # TODO: confirm this matches ADiT's actual Data/Batch object.
            if hasattr(batch, "batch") and hasattr(batch, "num_graphs"):
                graph_index = batch.batch  # shape [num_atoms], values in [0, num_graphs)
                num_atoms = int(graph_index.numel())
                num_samples = int(batch.num_graphs)
                atoms_per_sample = graph_index.bincount(minlength=num_samples)
                max_atoms_per_sample = int(atoms_per_sample.max().item()) if num_samples else 0

                # TODO: replace with the real token field if tokens != atoms
                # in ADiT (e.g. `batch.tokens`, `batch.input_ids`, a separate
                # sequence modality, etc.)
                num_tokens = getattr(batch, "num_tokens", None)
                if num_tokens is not None:
                    num_tokens = int(num_tokens.sum().item()) if hasattr(num_tokens, "sum") else int(num_tokens)
                    max_tokens_per_sample = int(getattr(batch, "num_tokens").max().item())
                else:
                    num_tokens = num_atoms
                    max_tokens_per_sample = max_atoms_per_sample

                return {
                    "num_atoms": num_atoms,
                    "num_tokens": num_tokens,
                    "num_samples": num_samples,
                    "max_atoms_per_sample": max_atoms_per_sample,
                    "max_tokens_per_sample": max_tokens_per_sample,
                }

            # --- 2. dict-style batch ---------------------------------------
            if isinstance(batch, dict):
                atom_key = next((k for k in ("num_atoms", "n_atoms", "atom_types", "positions") if k in batch), None)
                token_key = next((k for k in ("num_tokens", "n_tokens", "input_ids", "tokens") if k in batch), None)

                num_atoms = _count_from_field(batch.get(atom_key)) if atom_key else None
                num_tokens = _count_from_field(batch.get(token_key)) if token_key else num_atoms

                if num_atoms is None:
                    return None

                num_samples = _infer_num_samples(batch)
                return {
                    "num_atoms": num_atoms,
                    "num_tokens": num_tokens if num_tokens is not None else num_atoms,
                    "num_samples": num_samples,
                    "max_atoms_per_sample": num_atoms // max(num_samples, 1),
                    "max_tokens_per_sample": (num_tokens or num_atoms) // max(num_samples, 1),
                }

            # --- 3. plain object with attributes ---------------------------
            for atom_attr in ("num_atoms", "n_atoms", "atom_types", "positions"):
                if hasattr(batch, atom_attr):
                    num_atoms = _count_from_field(getattr(batch, atom_attr))
                    if num_atoms is not None:
                        num_samples = getattr(batch, "num_graphs", None) or getattr(batch, "batch_size", 1)
                        return {
                            "num_atoms": num_atoms,
                            "num_tokens": num_atoms,
                            "num_samples": int(num_samples),
                            "max_atoms_per_sample": num_atoms // max(int(num_samples), 1),
                            "max_tokens_per_sample": num_atoms // max(int(num_samples), 1),
                        }

            return None
        except Exception as e:
            log.warning(f"BatchSizeMonitor: extraction error: {e}")
            return None


def _count_from_field(field: Any) -> Optional[int]:
    """Turn a tensor/list/int field into a total element count."""
    if field is None:
        return None
    if hasattr(field, "numel"):  # torch.Tensor
        return int(field.numel() if field.dim() <= 1 else field.shape[0])
    if hasattr(field, "__len__"):
        return len(field)
    if isinstance(field, int):
        return field
    return None


def _infer_num_samples(batch: dict) -> int:
    for key in ("num_graphs", "batch_size", "ptr"):
        if key in batch:
            val = batch[key]
            if hasattr(val, "numel"):
                return int(val.numel() - 1) if key == "ptr" else int(val)
            return int(val)
    return 1
