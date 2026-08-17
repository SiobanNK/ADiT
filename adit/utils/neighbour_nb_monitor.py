import torch
import matplotlib
matplotlib.use("Agg")  # pas de display sur les nœuds de calcul Jean-Zay
import matplotlib.pyplot as plt
from torch_geometric.utils import degree
from pytorch_lightning import Callback

from adit.models.net.adit.utils import generate_euclidian_edge_index, generate_token_coordinates  # adapte le chemin


class NeighbourStatsCallback(Callback):
    """
    Callback de debug : calcule les stats (mean/max/min/std/median) du nombre
    de voisins par atome/token, à partir d'un batch, et sauvegarde un histogramme.

    Actif uniquement si `enabled=True`, et déclenché tous les `every_n_steps` steps.
    """

    def __init__(
        self,
        atom_neighbour_radius: float,
        token_neighbour_radius: float,
        enabled: bool = False,
        every_n_steps: int = 500,
        output_dir: str = "neighbour_stats",
    ):
        super().__init__()
        self.atom_neighbour_radius = atom_neighbour_radius
        self.token_neighbour_radius = token_neighbour_radius
        self.enabled = enabled
        self.every_n_steps = every_n_steps
        self.output_dir = output_dir

        if self.enabled:
            import os
            os.makedirs(self.output_dir, exist_ok=True)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self.enabled:
            return
        if trainer.global_step % self.every_n_steps != 0:
            return

        atom_mask = batch["atom_mask"].bool()
        token_mask = batch["seq_mask"].bool()
        if self.remove_protein_ligand_edge:
            atom_mask = atom_mask[token_mask]   # (num_token, 37)

        atom_coordinates = batch["atom_positions"][atom_mask]
        num_tokens = batch["seq_mask"].sum(dim=-1)
        num_atoms = batch["atom_mask"].sum(-1)[token_mask].int()

        step_tag = f"step{trainer.global_step}"

        if self.atom_neighbour_radius > 0.0:
            atom_edges = generate_euclidian_edge_index(
                num_atoms, atom_coordinates, self.atom_neighbour_radius
            )
            self._log_stats(
                atom_edges, num_atoms.sum().item(),
                name=f"atom_{step_tag}", trainer=trainer,
            )

        if self.token_neighbour_radius > 0.0:
            atom2token = torch.arange(num_tokens.sum(), device=batch["aatype"].device).repeat_interleave(num_atoms)
            token_coordinates = generate_token_coordinates(atom_coordinates, atom2token)
            token_edges = generate_euclidian_edge_index(
                num_tokens, token_coordinates, self.token_neighbour_radius
            )
            self._log_stats(
                token_edges, num_tokens.sum().item(),
                name=f"token_{step_tag}", trainer=trainer,
            )

    def _log_stats(self, edge_index, num_nodes_total, name, trainer):
        if edge_index is None or edge_index.numel() == 0:
            print(f"[{name}] pas d'arêtes (graphe vide)")
            return

        deg = degree(edge_index[0], num_nodes=num_nodes_total).float()

        stats = {
            "mean":   deg.mean().item(),
            "max":    deg.max().item(),
            "min":    deg.min().item(),
            "std":    deg.std().item(),
            "median": deg.median().item(),
        }

        print(
            f"[{name}] n_nodes={num_nodes_total} | "
            f"mean={stats['mean']:.2f}  max={stats['max']:.0f}  "
            f"min={stats['min']:.0f}  std={stats['std']:.2f}  median={stats['median']:.1f}"
        )

        # log dans le logger (TensorBoard/W&B) si dispo
        if trainer.logger is not None:
            for k, v in stats.items():
                trainer.logger.log_metrics({f"neighbour_stats/{name}_{k}": v}, step=trainer.global_step)

        # histogramme
        deg_np = deg.cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = int(deg_np.max() - deg_np.min() + 1)
        ax.hist(deg_np, bins=bins, edgecolor="black", alpha=0.75)
        ax.axvline(stats["mean"], color="red", linestyle="--", label=f"mean={stats['mean']:.1f}")
        ax.axvline(stats["median"], color="green", linestyle="--", label=f"median={stats['median']:.1f}")
        ax.set_xlabel("Nombre de voisins")
        ax.set_ylabel("Nombre de nœuds")
        ax.set_title(f"Distribution des voisins — {name}")
        ax.legend()
        fig.tight_layout()

        save_path = f"{self.output_dir}/{name}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
