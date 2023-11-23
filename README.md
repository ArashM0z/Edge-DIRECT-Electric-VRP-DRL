# EFECTIW-ROTER — Heterogeneous Fleet & Demand VRP with Time-Window Constraints

> **Forked from [wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route)** (Kool et al., ICLR 2019). This repo implements the **EFECTIW-ROTER** architecture introduced in our ACM SIGSPATIAL 2024 paper.

[![Paper](https://img.shields.io/badge/ACM%20SIGSPATIAL-2024-blue)](https://doi.org/10.1145/3678717.3691208)
[![Forked from](https://img.shields.io/badge/forked%20from-wouterkool/attention--learn--to--route-lightgrey)](https://github.com/wouterkool/attention-learn-to-route)

## What EFECTIW-ROTER is

EFECTIW-ROTER = **spatial Edge-Feature EnhanCed mulTIgraph fusion encoder With spectral-based embedding and hieRarchical decOder with learnable TEmpoRal positional embedding**.

It solves the **Heterogeneous Fleet & Demand VRP with Time-Window Constraints (HFDVRPTW)** with DRL on top of the Kool 2019 Attention Model.

## What this fork adds

| File | What |
|---|---|
| `problems/hf_vrptw/graphs.py` | Two sparse graphs: TW-feasibility `G_tw` and demand-vehicle compatibility `G_dv` |
| `problems/hf_vrptw/spectral_embedding.py` | Bottom-k Laplacian eigenvectors as per-node spectral features |
| `nets/efectiw/spatial_encoder.py` | Spatial-Edge-Feature graph Transformer (attention bias from travel-time edges, masked by `G_tw`) |
| `nets/efectiw/temporal_encoder.py` | Temporal Graph Transformer with **learnable temporal positional embedding** (rank by `tw_start`), masked by `G_dv` |
| `nets/efectiw/fusion_encoder.py` | Multigraph fusion: gated combination of spatial + temporal + spectral |
| `nets/efectiw/hierarchical_decoder.py` | Hierarchical decoder: vehicle-type selector then node selector |
| `nets/efectiw/efectiw_model.py` | Full agent composing all four modules above |

The Kool 2019 training loop, REINFORCE rollout baseline, and `run.py` entry point are unchanged.

## Run

```bash
python run.py --problem hf_vrptw --graph_size 50 --baseline rollout --run_name efectiw-n50
```

## Citation

```bibtex
@inproceedings{mozhdehi2024efectiwroter,
  title={{EFECTIW-ROTER}: Deep Reinforcement Learning Approach for Solving Heterogeneous Fleet and Demand VRP With Time-Window Constraints},
  author={Mozhdehi, Arash and Mohammadizadeh, Mahdi and Wang, Yunli and Sun, Sun and Wang, Xin},
  booktitle={ACM SIGSPATIAL 2024},
  pages={17--28},
  year={2024},
  doi={10.1145/3678717.3691208}
}
@inproceedings{kool2019attention,
  title={Attention, Learn to Solve Routing Problems!},
  author={Kool, Wouter and van Hoof, Herke and Welling, Max},
  booktitle={ICLR},
  year={2019}
}
```
