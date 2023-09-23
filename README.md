# Edge-DIRECT — Heterogeneous Electric VRP with Time Windows via DRL

> **Forked from [wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route)** (Kool et al., ICLR 2019). This repo implements **Edge-DIRECT** introduced in our Canadian AI 2024 paper.

[![Paper](https://img.shields.io/badge/Canadian%20AI-2024-blue)](https://caiac.pubpub.org/pub/vlg4rwhi)
[![arXiv](https://img.shields.io/badge/arXiv-2407.01615-b31b1b)](https://arxiv.org/abs/2407.01615)
[![Forked from](https://img.shields.io/badge/forked%20from-wouterkool/attention--learn--to--route-lightgrey)](https://github.com/wouterkool/attention-learn-to-route)

## Edge-DIRECT in one line

**E**dge-enhanced **D**ual att**I**ntion enco**R**der and feature-**E**nhan**C**ed dual a**T**tention decoder.

## What this fork adds on top of Kool 2019

| File | What |
|---|---|
| `problems/evrptw/graphs.py` | The **extra graph representation** — adjacency from time-window overlap between customers |
| `problems/evrptw/charging.py` | Non-linear CC + CV charging model + energy-consumption helper |
| `nets/edge_direct/dual_attention_encoder.py` | **Dual-attention encoder** layered as (spatial-edge attention + energy-edge attention) per layer, gated + merged |
| `nets/edge_direct/dual_attention_decoder.py` | **Feature-enhanced dual decoder**: vehicle-type head + node head with SOC / capacity / time context |
| `nets/edge_direct/edge_direct_model.py` | Full agent composing the encoder + decoder |
| `README.md` | This file |

The encoder is "edge-enhanced" because every attention block carries one spatial-edge bias (travel-time) and one energy-edge bias. The decoder is "feature-enhanced" because its context vector includes SOC and remaining capacity, beyond the usual last-node / graph embedding.

## Run

```bash
python run.py --problem evrptw --graph_size 30 --baseline rollout --run_name edge-direct-n30
```

## Citation

```bibtex
@inproceedings{mozhdehi2024edgedirect,
  title={{Edge-DIRECT}: A Deep Reinforcement Learning-based Method for Solving Heterogeneous Electric VRP with Time Window Constraints},
  author={Mozhdehi, Arash and Mohammadizadeh, Mahdi and Wang, Xin},
  booktitle={Canadian Conference on Artificial Intelligence},
  year={2024}
}
@inproceedings{kool2019attention,
  title={Attention, Learn to Solve Routing Problems!},
  author={Kool, Wouter and van Hoof, Herke and Welling, Max},
  booktitle={ICLR},
  year={2019}
}
```
