"""
Sonde les tailles de batch (nb de tokens, nb d'atomes) d'ADiT sans jamais
calculer de gradient et, en option, sans même appeler le modèle.

Idée générale :
  1) On instancie uniquement le LightningDataModule (via Hydra, comme le fait
     train.sh) et on itère sur le dataloader d'entraînement. Aucun forward,
     aucun backward -> zéro calcul de gradient, zéro besoin de GPU.
  2) En option (--forward), on instancie aussi le modèle et on fait un
     forward sous torch.no_grad() pour vérifier les tailles après
     featurisation/tokenisation, toujours sans backward.

Usage (à adapter selon vos overrides Hydra, cf. train.sh) :

    python probe_batch_sizes.py experiment=lba_S ++data.batch_size=16 \
        ++data.dataset.split_identity_threshold=identity_30 \
        n_batches=20

    # avec un forward pass en plus (no_grad) :
    python probe_batch_sizes.py experiment=lba_S ckpt_path=ckpts/adit_S.ckpt \
        forward=true n_batches=5
"""

import sys
import torch
import hydra
import rootutils
from omegaconf import DictConfig

# Aligne le comportement de résolution de chemins avec train.sh / test.sh
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def tensor_stats(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)} dtype={t.dtype}"


def walk(obj, prefix="batch", max_items=200, seen=0):
    """Parcourt récursivement un batch (dict / objet type PyG Data /
    liste / tuple) et affiche la forme de chaque tenseur trouvé."""
    if seen > max_items:
        return seen

    if torch.is_tensor(obj):
        print(f"  {prefix}: {tensor_stats(obj)}")
        return seen + 1

    if isinstance(obj, dict):
        for k, v in obj.items():
            seen = walk(v, f"{prefix}.{k}", max_items, seen)
        return seen

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            seen = walk(v, f"{prefix}[{i}]", max_items, seen)
        return seen

    # Objets type torch_geometric.data.Data / Batch : ils exposent .keys / .items()
    keys_attr = getattr(obj, "keys", None)
    if callable(keys_attr):
        try:
            for k in obj.keys():
                seen = walk(getattr(obj, k), f"{prefix}.{k}", max_items, seen)
            return seen
        except TypeError:
            pass

    return seen


TOKEN_HINTS = ("token", "residue", "res_")
ATOM_HINTS = ("atom",)


def guess_counts(obj, hints, prefix="batch"):
    """Essaie de deviner le nb de tokens/atomes en cherchant des clés dont
    le nom contient un des mots-clés (token/atom/residue), et renvoie la
    plus grande dimension "longueur de séquence" trouvée (en excluant la
    dimension batch=0)."""
    found = []

    def _rec(o, path):
        if torch.is_tensor(o):
            name = path.lower()
            if any(h in name for h in hints) and o.dim() >= 1:
                # dimension la plus probable = la plus grande hors dim 0 (batch)
                dims = list(o.shape)
                found.append((path, dims))
            return
        if isinstance(o, dict):
            for k, v in o.items():
                _rec(v, f"{path}.{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                _rec(v, f"{path}[{i}]")
        else:
            keys_attr = getattr(o, "keys", None)
            if callable(keys_attr):
                try:
                    for k in o.keys():
                        _rec(getattr(o, k), f"{path}.{k}")
                except TypeError:
                    pass

    _rec(obj, prefix)
    return found


@hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    n_batches = int(cfg.get("n_batches", 10))
    do_forward = bool(cfg.get("forward", False))

    print(f"[probe] instanciation du datamodule ({cfg.data._target_})")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="fit")
    loader = datamodule.train_dataloader()

    model = None
    if do_forward:
        print(f"[probe] instanciation du modèle ({cfg.model._target_}) — forward en no_grad, jamais de backward")
        model = hydra.utils.instantiate(cfg.model)
        model.eval()
        if cfg.get("ckpt_path"):
            state = torch.load(cfg.ckpt_path, map_location="cpu")
            state_dict = state.get("state_dict", state)
            model.load_state_dict(state_dict, strict=False)

    n_token_max, n_atom_max = 0, 0

    with torch.no_grad():  # ceinture + bretelles : aucun graphe de calcul construit
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break

            print(f"\n=== batch {i} ===")
            walk(batch)

            token_hits = guess_counts(batch, TOKEN_HINTS)
            atom_hits = guess_counts(batch, ATOM_HINTS)

            if token_hits:
                print("  -> candidats 'token':", [(p, d) for p, d in token_hits])
                n_token_max = max(n_token_max, *(max(d) for _, d in token_hits))
            if atom_hits:
                print("  -> candidats 'atom':", [(p, d) for p, d in atom_hits])
                n_atom_max = max(n_atom_max, *(max(d) for _, d in atom_hits))

            if model is not None:
                out = model(batch)  # forward seul, pas de loss.backward()
                print("  -> sortie modèle:")
                walk(out, prefix="output")

    print(f"\n[probe] max token-dim observée sur {n_batches} batches : {n_token_max}")
    print(f"[probe] max atom-dim observée sur {n_batches} batches : {n_atom_max}")
    print("[probe] Vérifiez visuellement la liste ci-dessus : les noms de clés ")
    print("        exacts dépendent de votre datamodule (ex: token_pad_mask, ")
    print("        atom_pad_mask, ref_pos, num_atoms, ptr, etc.)")


if __name__ == "__main__":
    sys.exit(main())
