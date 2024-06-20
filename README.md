# SED2AM — Multi-Trip Time-Dependent VRP via Deep RL

> **Forked from [wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route)** (Kool et al., ICLR 2019). This repo adds the **multi-trip time-dependent VRP** variant introduced in our ACM TKDD 2025 paper.

[![Paper](https://img.shields.io/badge/ACM%20TKDD-2025-blue)](https://doi.org/10.1145/3721983)
[![Forked from](https://img.shields.io/badge/forked%20from-wouterkool/attention--learn--to--route-lightgrey)](https://github.com/wouterkool/attention-learn-to-route)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What this fork adds

The Kool 2019 codebase ships with `tsp`, `cvrp`, `sdvrp`, `op`, `pctsp_det`, `pctsp_stoch` problem definitions. This fork adds **`mttdvrp`** — a new problem registered through the same conventions:

| File | Status | What it does |
|---|---|---|
| `problems/mttdvrp/problem_mttdvrp.py` | new | MTTDVRP problem with vectorised piecewise-linear time-dependent travel-time cost |
| `problems/mttdvrp/state_mttdvrp.py`  | new | State container tracking `current_time` and the multi-trip-aware feasibility mask |
| `problems/__init__.py`               | patch | Register `MTTDVRP` in the problem registry |
| `options.py`                         | patch | Add `mttdvrp` to the `--problem` help string |
| `utils/functions.py`                 | patch | Add the `mttdvrp` entry to `PROBLEM_REGISTRY` |
| `README.md`                          | rewritten | This file |

Everything else — the attention encoder, the pointer decoder, REINFORCE with rollout baseline, the run/train scripts — is **unchanged** from Kool 2019.

## Two modifications, end to end

1. **Multi-trip**: a vehicle may visit the depot mid-tour, refilling its capacity and continuing to serve customers. Kool's `pi` tour representation already encodes depot returns; the change is in how cost and the feasibility mask are computed.
2. **Time-dependent travel**: travel time between any two nodes is the integral of distance over a piecewise-linear speed profile *starting at the leg's departure time*. Each instance carries its own profile (breakpoints + per-segment speeds).

Both modifications are in the new `problems/mttdvrp/` directory. The training loop is unchanged.

## Run

```bash
python run.py --problem mttdvrp --graph_size 50 --baseline rollout --run_name sed2am-n50
```

CLI flags inherit from the Kool framework: see `options.py`.

## Citation

If you use this fork, please cite both the Kool 2019 base and our paper:

```bibtex
@article{mozhdehi2025sed2am,
  title={{SED2AM}: Solving Multi-Trip Time-Dependent Vehicle Routing Problem Using Deep Reinforcement Learning},
  author={Mozhdehi, Arash and Wang, Yunli and Sun, Sun and Wang, Xin},
  journal={ACM Transactions on Knowledge Discovery from Data},
  year={2025},
  doi={10.1145/3721983}
}
@inproceedings{kool2019attention,
  title={Attention, Learn to Solve Routing Problems!},
  author={Kool, Wouter and van Hoof, Herke and Welling, Max},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```

## License

MIT, inheriting from the Kool 2019 base.
