"""Precompute the length (L = token count) cache used by DynamicBatchSampler.

Run this once (e.g. in an interactive `srun` session or a small 1-CPU job)
BEFORE launching the actual multi-GPU/multi-node training job, so every rank
just reads the cache instead of racing to (re)compute and write it on Jean-
Zay's shared filesystem.

Example (adapt the Hydra overrides to your experiment config):

    python scripts/precompute_lengths.py experiment=lba_S \
        ++data.dataset.split_identity_threshold=identity_30

This instantiates `cfg.data.dataset` and `cfg.data` exactly like train.sh
would, then calls `setup("fit")` and `setup("test")` so both the
train/val split and the test set get a length cache written to
`cfg.data.length_cache_dir`.
"""
import rootutils
import hydra
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    assert cfg.data.get("length_cache_dir") is not None, (
        "Set data.length_cache_dir (e.g. ++data.length_cache_dir=lengths/lba_S_identity_30) "
        "so the cache written here is the one your real training job will read."
    )

    datamodule = hydra.utils.instantiate(cfg.data)

    datamodule.setup(stage="fit")
    datamodule._lengths_for(datamodule.data_train, split="train")
    datamodule._lengths_for(datamodule.data_val, split="val")

    datamodule.setup(stage="test")
    datamodule._lengths_for(datamodule.data_test, split="test")

    print(f"Cached lengths under {cfg.data.length_cache_dir}")


if __name__ == "__main__":
    main()
