# EFECTIW-ROTER — Heterogeneous Fleet VRP with Time Windows via Deep RL

> **Forked from [wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route)** (Kool et al., ICLR 2019). This repo adds the **heterogeneous-fleet VRP with time windows** variant introduced in our ACM SIGSPATIAL 2024 paper.

[![Paper](https://img.shields.io/badge/ACM%20SIGSPATIAL-2024-blue)](https://doi.org/10.1145/3678717.3691208)
[![Forked from](https://img.shields.io/badge/forked%20from-wouterkool/attention--learn--to--route-lightgrey)](https://github.com/wouterkool/attention-learn-to-route)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What this fork adds

The Kool 2019 codebase covers `tsp`, `cvrp`, `sdvrp`, `op`, `pctsp_det`, `pctsp_stoch`. This fork adds **`hf_vrptw`**:

| File | Status | What |
|---|---|---|
| `problems/hf_vrptw/problem_hf_vrptw.py` | new | HF-VRPTW problem with per-vehicle cost-per-km, fixed costs, and TW slack penalty |
| `problems/hf_vrptw/state_hf_vrptw.py` | new | State container tracking `arrival_t` and per-vehicle capacity |
| `problems/__init__.py` | patch | Register `HFVRPTW` |
| `options.py` | patch | Add `hf_vrptw` to the `--problem` help string |
| `README.md` | rewritten | This file |

The encoder, decoder, REINFORCE training loop, and run script are **unchanged** from Kool 2019.

## Two modifications

1. **Heterogeneous fleet**: a fleet of four vehicle types (capacity, cost-per-km, fixed cost) parameterised in `VEHICLE_FLEET`. Each instance is pre-assigned a vehicle type; the cost function applies the corresponding per-km and fixed costs.
2. **Hard time windows**: every customer carries `[tw_start, tw_end]`. Arrivals wait until `tw_start`, and a per-minute slack penalty is added to the cost for arrivals after `tw_end`. Service takes 10 minutes.

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
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```
