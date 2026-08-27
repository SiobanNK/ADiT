# ADiT
This repo is based on the codebase of the paper **Towards All-Atom Foundation Models for Biomolecular Binding Affinity Prediction**, *ICLR'2026*
[[OpenReview](https://openreview.net/forum?id=o0Qfsq1fK8)].

The official codebase can be found at [[GitHub](https://github.com/VectorShi/ADiT)].

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

After trying Graph Convolution and Graph Attention, we retained Graph Attention with an RBF embedding of the distances between nodes.

We inserted each Graph Attention Layer after Diffusion Transformer building the main trunk of ADiT.


## Reproduction



<!-- ## Citation -->
