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


### Distance embedding in token pair conditioning

We firstly added structure information to the token-level pair conditioning.

Goal: introduire info 3D explicite dans token pair rep.

### Structural graph attention

The second improvement consists in updating the single representations of atoms (res. tokens) based on the closest atoms
(resp. tokens) in the 3D molecule structure, by adding message passing.

Indeed, the pair and single conditioning representations are originally based on the sequence :
the token pair representation is dense within each batch item, while the atom pair representation links groups of 32 atoms with the surronding 128 atoms in sequence.
Our message passing update considers a graph in which neighbourhood is determined on geometric distance rather than sequential.
After trying Graph Convolution and Graph Attention (GAT) at different positions in ADiT main trunk, we retained Graph Attention inserted after each Diffusion Transformer block. An RBF embedding of the distances between neighbours is given to the GAT.


## Reproduction

### Distance embedding in token pair conditioning

```bash
bash train.sh experiment=new_denoise_S_pdb_fixed_0_5 ++model.net.token_coord_encoder='rbf' ++model.net.relative_position_d_max=2.2 ++trainer.devices=4 trainer.min_epochs=33 trainer.max_epochs=33
```

The resulting checkpoints are automatically saved in `outputs/YYYY-MM-DD/denoising_S_scale0_5/checkpoints`. Take the last one (`last.ckpt`) for the finetuning. The bash instructions are identical to the original codebase, with the additional arguments `++model.net.token_coord_encoder='rbf' ++model.net.relative_position_d_max=2.2` to the distance embedding. `relative_position_d_max` must be given in nm and corresponds to the highest center of the RBF embedding.

If `model.net.token_coord_encoder` is not precised, default value is `null`, corresponding to the original ADiT model.

You can experiment with `++model.net.token_coord_encoder='onehot'`.


### STructural Graph Attention

```bash
bash train.sh experiment=new_denoise_S_pdb_fixed_0_5 +model/gnn='gat' ++trainer.devices=4 trainer.min_epochs=33 trainer.max_epochs=33
```

The resulting checkpoints are automatically saved in `outputs/YYYY-MM-DD/denoising_S_scale0_5/checkpoints`. Take the last one (`last.ckpt`) for the finetuning. The bash instructions are identical to the original codebase, with the additional argument `+model/gnn='gat'` to add geometric graph attention after the diffusion transformers.



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
- Zhang, Xu, Chenthamarakshan, Lozano, Das, Tang. Enhancing protein language models with structure based encoder and pre-training. ICLR MLDD Workshop, 2023
- Brody, Alon, Yahav. How attentive are graph attention networks? ICLR, 2022
- K. T. Schütt, P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko3, K.-R. Müller. SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. NIPS 2017.
