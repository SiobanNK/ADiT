"""
Sonde la distribution du nombre de voisins (degré) par atome et par token
sur plusieurs batches d'ADiT, sans jamais calculer de gradient ni instancier
le modèle : le edge_index ne dépend que des coordonnées + masks du batch, on
peut donc réutiliser directement generate_euclidian_edge_index /
generate_token_coordinates.

Usage :

    python scripts/attention_stats.py experiment=lba_S ++data.batch_size=16 \
        +model.net.atom_neighbour_radius=0.5 +model.net.token_neighbour_radius=1.0 +n_batches=20

ATTENTION ! Les coordonnées sont converties en nm (cf config).

Si atom_neighbour_radius / token_neighbour_radius ne sont pas passés en
argument, on essaie de les lire depuis cfg.model (mêmes noms que dans le
forward du modèle).
"""

import os
from pathlib import Path
import sys
import torch
import hydra
import rootutils
from omegaconf import DictConfig
import numpy as np
from torch_geometric.utils import degree

import matplotlib
matplotlib.use("Agg")  # pas de display sur les noeuds de calcul Jean-Zay
import matplotlib.pyplot as plt

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from adit.models.net.adit.utils import generate_euclidian_edge_index, generate_token_coordinates  # noqa: E402


def compute_degrees(edge_index, num_nodes_total):
    """Renvoie le tenseur des degrés (1 valeur par noeud), ou None si pas d'arêtes."""
    if edge_index is None or edge_index.numel() == 0:
        return None
    return degree(edge_index[0], num_nodes=num_nodes_total).float()


def print_stats(deg: torch.Tensor, name: str):
    if deg is None or deg.numel() == 0:
        print(f"[{name}] aucune donnée")
        return
    print(
        f"[{name}] n_nodes={deg.numel()} | "
        f"mean={deg.mean().item():.2f}  max={deg.max().item():.0f}  "
        f"min={deg.min().item():.0f}  std={deg.std().item():.2f}  "
        f"median={deg.median().item():.1f}"
    )

def save_figure(all_atom_degrees, all_token_degrees, dataset: str, atom_neighbour_radius, token_neighbour_radius, output_dir: str):
    fig, axes = plt.subplots(2,figsize=(6, 8))
    fig.suptitle(f"Distribution des voisins — {dataset} (tous batches confondus)\nAtom radius = {atom_neighbour_radius} nm ; token radius = {token_neighbour_radius} nm")

    if all_atom_degrees:
        atom_degrees = torch.cat(all_atom_degrees)
        print_stats(atom_degrees, "atom")
        save_histogram(axes[0], atom_degrees, "atom")
    else:
        print("[atom] atom_neighbour_radius <= 0 ou aucune arête -> rien à calculer")

    if all_token_degrees:
        token_degrees = torch.cat(all_token_degrees)
        print_stats(token_degrees, "token")
        save_histogram(axes[1], token_degrees, "token")
    else:
        print("[token] token_neighbour_radius <= 0 ou aucune arête -> rien à calculer")

    fig.tight_layout()

    save_path = f"{output_dir}/{dataset}_neighbour_hist.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[{dataset}] histogramme sauvegardé -> {save_path}")


def save_histogram(ax, deg: torch.Tensor, name: str, max_ticks: int = 7):
    deg_np = deg.cpu().numpy()
    mean, median = deg_np.mean(), float(torch.tensor(deg_np).median())

    lo, hi = int(deg_np.min()), int(deg_np.max())
    bin_edges = np.arange(lo - 0.5, hi + 1.5, 1)  # un bin par entier, centré sur k

    ax.hist(deg_np, bins=bin_edges, edgecolor="black", alpha=0.75)

    # au plus max_ticks ticks, répartis uniformément sur [lo, hi] et arrondis
    # à des entiers pour rester cohérent avec des comptes de voisins
    n_ticks = min(max_ticks, hi - lo + 1)
    ticks = np.unique(np.round(np.linspace(lo, hi, n_ticks)).astype(int))
    ax.set_xticks(ticks)

    ax.axvline(mean, color="red", linestyle="--", label=f"mean={mean:.1f}")
    ax.axvline(median, color="green", linestyle="--", label=f"median={median:.1f}")
    ax.set_xlabel("Nombre de voisins")
    ax.set_ylabel("Nombre de nœuds")
    ax.set_title(name)
    ax.legend()


@hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    n_batches = int(cfg.get("n_batches", 10))
    output_dir = cfg.get("output_dir", "./outputs/GCN_neighbours_stats")
    os.makedirs(output_dir, exist_ok=True)

    atom_neighbour_radius = float(
        cfg.get("atom_neighbour_radius", cfg.model.net.get("atom_neighbour_radius", 0.0))
    )
    token_neighbour_radius = float(
        cfg.get("token_neighbour_radius", cfg.model.net.get("token_neighbour_radius", 0.0))
    )
    print(f"[probe] atom_neighbour_radius={atom_neighbour_radius}  token_neighbour_radius={token_neighbour_radius}")

    print(f"[probe] instanciation du datamodule ({cfg.data._target_})")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="fit")
    loader = datamodule.train_dataloader()

    all_atom_degrees = []
    all_token_degrees = []

    with torch.no_grad():  # pas de graphe de calcul construit
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break

            atom_mask = batch["atom_mask"].bool()
            token_mask = batch["seq_mask"].bool()

            atom_coordinates = batch["atom_positions"][atom_mask]
            num_tokens = batch["seq_mask"].sum(dim=-1)
            # num_atoms = batch["atom_mask"].sum(-1)[token_mask].int() # nb per token, not molecule !
            num_atoms = batch["atom_mask"].sum(dim=(1,2)).int()

            if atom_neighbour_radius > 0.0:
                atom_edges = generate_euclidian_edge_index(
                    num_atoms, atom_coordinates, atom_neighbour_radius
                )
                deg = compute_degrees(atom_edges, num_atoms.sum().item())
                if deg is not None:
                    all_atom_degrees.append(deg)

            if token_neighbour_radius > 0.0:
                num_atoms_per_token = batch["atom_mask"].sum(-1)[token_mask].int()
                atom2token = torch.arange(num_tokens.sum(), device='cpu').repeat_interleave(num_atoms_per_token)
                token_coordinates = generate_token_coordinates(atom_coordinates, atom2token)
                print(token_coordinates)
                token_edges = generate_euclidian_edge_index(
                    num_tokens, token_coordinates, token_neighbour_radius
                )
                deg = compute_degrees(token_edges, num_tokens.sum().item())
                if deg is not None:
                    all_token_degrees.append(deg)

            print(f"[probe] batch {i} traité (n_atoms={num_atoms.sum().item()}, n_tokens={num_tokens.sum().item()})")

    print(f"\n[probe] === stats agrégées sur {n_batches} batches ===")

    dataset_path = Path(cfg.data.dataset.get("path_to_dataset"))
    dataset = dataset_path.name
    save_figure(all_atom_degrees, all_token_degrees, dataset, atom_neighbour_radius, token_neighbour_radius, output_dir)


if __name__ == "__main__":
    sys.exit(main())
