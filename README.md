# ADiT
This repo is based on the codebase of the paper **Towards All-Atom Foundation Models for Biomolecular Binding Affinity Prediction**, *ICLR'2026*
[[OpenReview](https://openreview.net/forum?id=o0Qfsq1fK8)].

The official original codebase can be found at [[GitHub](https://github.com/VectorShi/ADiT)].

## Overview

Binding affinity between biomolecules is a key indicator in the drug discovery process and other other molecular interactions. The Atom-level Diffusion Transformer (ADiT), a recent foundation model, managed to predict binding affinity with state-of-the-art or competitive results by jointly embedding sequence and structural information. While ADiT encodes 3D geometry in input, its multi-scale attention mechanisms miss long-range dependencies between atoms or residues distant in sequence but close in structure due to molecule folding. In this work, we complete ADiT with structural atom and residue distance representations, by adding light layers of embedding and geometric graph attention. We firstly added explicit geometric distance to the token-level conditioning, and then inserted graph attention layers within the main trunk consisting in a stack of atom and token-level Diffusion Transformers. The resulting upgraded model achieves better prediction results compared to the original ADiT, with neglectable additional parameters and training time.

## Installation
Refer to the original codebase to install the dependencies, checkpoints and datasets.

## Changes upon the original codebase

The following diagram is based on the figure 1 of the original ADiT paper.

We added to ADiT the graph attention blocks (red).

![ADiT](./asset/GAT.svg)




### Structural distance embedding in token pair conditioning

We firstly added structure information to the token-level pair conditioning, experimenting with RBF and onehot encoding. The RBF representation spans 0.2 nm to 2.2 nm with 64 bins.

See the file `adit/models/net/adit/relative_position_encoder.py` where we concatenate the 3D distance representation to the original sequential distance embedding.

### Structural graph attention

The second improvement consists in updating the single representations of atoms (res. tokens) based on the closest atoms
(resp. tokens) in the 3D molecule structure, by adding message passing.

Indeed, the pair and single conditioning representations are originally based on the sequence :
the token pair representation is dense within each batch item, while the atom pair representation links groups of 32 atoms with the surronding 128 atoms in sequence.
Our message passing update considers a graph in which neighbourhood is determined on geometric distance rather than sequential.
After trying Graph Convolution and Graph Attention (GAT) at different positions in ADiT main trunk, we retained Graph Attention inserted after each Diffusion Transformer block. An RBF embedding of the distances between neighbours is given to the GAT.

While atom coordinates are provided in input of ADiT, we chose to compute residues coordinates as the $C_\beta$ coordinates (or $C_\alpha$ in the case of glycine).

Let $\mathbf{x}$ the single representation of atoms (resp. tokens) in output of a Diffusion Transformer in ADiT main trunk. We update this representation $\mathbf{x}_i$ with the representations of the closest atoms to the atom (resp. token) $i$ and the embedding $\mathbf{e}$ of the distances between neighbours:

$$
\mathbf{x} = \text{LN}(\mathbf{x}) + \text{ReLU}(\text{GATv2Conv}(\text{LN}(\mathbf{x}),\mathbf{e}))
$$

such that GATv2Conv updates $\mathbf{y}_i$ as:

$$
\mathbf{y}_i' = \sum_{j\in N(i) \cup \{i\}} \alpha_{i,j} W_t y_j
$$

with the attention coefficients

$$
\alpha_{i,j} = \frac{\text{exp}(\mathbf{a}^T \text{LeakyReLU}(W_s\mathbf{y}_i + W_t\mathbf{y}_j + W_e\mathbf{e}_{i,j}))}{\sum_{k\in N(i) \cup \{i\}} \text{exp}(\mathbf{a}^T \text{LeakyReLU}(W_s\mathbf{y}_i + W_t\mathbf{y}_k + W_e\mathbf{e}_{i,k}))}
$$

where $W_s, W_t, W_e$ are learnable coefficients.


We defined the neighbours at atom-level by a maximum distance $d=0.5$ nm, and at token-level by $d=1$ nm. The edges are passed to GATv2Conv as an RBF embedding with 16 centers from 0 to $d$.

See the file `adit/models/net/adit/token_graph.py` for these additional layers and `adit/models/net/adit/utils.py` for the implementation of token coordinates and distance-based edges.


## Results


## Reproduction

### Structural distance embedding in token pair conditioning

```bash
bash train.sh experiment=new_denoise_S_pdb_fixed_0_5 ++model.net.token_coord_encoder='rbf' ++model.net.relative_position_d_max=2.2 ++trainer.devices=4 trainer.min_epochs=33 trainer.max_epochs=33
```

The resulting checkpoints are automatically saved in `outputs/YYYY-MM-DD/denoising_S_scale0_5/checkpoints`. Take the last one (`last.ckpt`) for the finetuning. The bash instructions are identical to the original codebase, with the additional arguments `++model.net.token_coord_encoder='rbf' ++model.net.relative_position_d_max=2.2` to the distance embedding. `relative_position_d_max` must be given in nm and corresponds to the highest center of the RBF embedding.

If `model.net.token_coord_encoder` is not precised, default value is `null`, corresponding to the original ADiT model.

You can experiment with `++model.net.token_coord_encoder='onehot'`.


### Structural Graph Attention

```bash
bash train.sh experiment=new_denoise_S_pdb_fixed_0_5 +model/gnn='gat' ++trainer.devices=4 trainer.min_epochs=33 trainer.max_epochs=33
```

The resulting checkpoints are automatically saved in `outputs/YYYY-MM-DD/denoising_S_scale0_5/checkpoints`. Take the last one (`last.ckpt`) for the finetuning. The bash instructions are identical to the original codebase, with the additional argument `+model/gnn='gat'` to add geometric graph attention after the diffusion transformers.

You can modify the neighbouring radius (in nm), the type of graph update layer and the type of residue coordinates in the configuration file `config/model/gnn/gat.yaml`.

## References

- Towards All-Atom Foundation Models for Biomolecular Binding Affinity Prediction, ICLR, 2026 [OpenReview]
- J. Jumper et al. Highly accurate protein structure prediction with alphafold. Nature, 2021
- Michael M. Bronstein, Joan Bruna, Taco Cohen, Petar Veličković. Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges. arXiv, 2021
- Wang, H. Liu, Y. Liu, Kurtin, Ji. Learning Hierarchical Protein Representations via Complete 3D Graph Networks. 2022
- Velickovic, Cucurull, Casanova, Romero, Lio, Bengio. Graph attention networks. ICLR, 2018
- Gilmer, Schoenholz, Riley, Vinyals, Dahl. Neural message passing for quantum chemistry. ICML, 2017
- Thomas Kipf & Max Welling. Semi-supervised classification with graph convolutional networks. ICLR, 2016
- MKearnes, S., McCloskey, K., Berndl, M. et al. Molecular graph convolutions: moving beyond fingerprints. J Comput Aided Mol Des 30, 595–608, 2016
- Robin Pearce & Yang Zhang. Deep learning techniques have significantly impacted protein structure prediction and protein design. Current opinion in structural biology, 2021
- Zhang, Xu, Chenthamarakshan, Lozano, Das, Tang. Enhancing  protein language models with structure based encoder and pre-training. ICLR MLDD Workshop, 2023
- Brody, Alon, Yahav. How attentive are graph attention networks? ICLR, 2022
- K. T. Schütt, P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko3, K.-R. Müller. SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. NIPS 2017.
