# SED2AM — Solving Multi-Trip Time-Dependent VRP with Deep RL

> **Forked from [wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route)** (Kool et al., ICLR 2019). This repo implements the **Simultaneous Encoder and Dual Decoder Attention Model** introduced in our ACM TKDD 2025 paper.

[![Paper](https://img.shields.io/badge/ACM%20TKDD-2025-blue)](https://doi.org/10.1145/3721983)
[![arXiv](https://img.shields.io/badge/arXiv-2503.04085-b31b1b)](https://arxiv.org/abs/2503.04085)
[![Forked from](https://img.shields.io/badge/forked%20from-wouterkool/attention--learn--to--route-lightgrey)](https://github.com/wouterkool/attention-learn-to-route)

## What SED2AM is

SED2AM solves the **Multi-Trip Time-Dependent Vehicle Routing Problem with maximum-working-hours constraints (MTTDVRP)** with DRL. It introduces three things over the Kool 2019 Attention Model:

1. **Temporal locality inductive bias in the encoder.** The graph is encoded **per time interval** — separate node and edge embeddings for each piece of the working day. Attention scores carry both an additive edge bias and a sigmoid edge gate, so traffic conditions modulate attention without inflating the parameter count.

2. **Dual decoder.** Each step picks a 2-tuple `(vehicle, next-location)`:
   - **Vehicle Selection Decoder** — given the fleet state `s_F^t` (a 5-tuple per vehicle: remaining capacity, current location, remaining working hours, current interval, time-left-in-interval) and the graph embedding, pick which vehicle expands its trip next.
   - **Trip Construction Decoder** — given the chosen vehicle's current interval, attend to the encoder output **at that interval** to pick the next location.

3. **Maximum working hours constraint.** Each vehicle has `τ_i^t` ≥ 0 remaining minutes; the feasibility mask refuses any move whose travel time would push it below zero.

The encoder, decoders, REINFORCE training loop with rollout baseline (Kool 2019), and run script are all wired together; only the encoder/decoder modules and the problem definition are new.

## File-level diff against the Kool 2019 base

| Path | Status | What |
|---|---|---|
| `nets/sed2am_encoder.py` | new | Simultaneous encoder, per-interval node + edge embeddings, edge-bias + edge-gate attention, gating mechanism (§4.2) |
| `nets/vehicle_selection_decoder.py` | new | Vehicle-selection decoder (§4.3.1) |
| `nets/trip_construction_decoder.py` | new | Trip-construction decoder (§4.3.2) |
| `nets/sed2am_model.py` | new | SED2AM agent composing encoder + dual decoder; action is `(i, v_j)` |
| `problems/mttdvrp/problem_mttdvrp.py` | new | MTTDVRP problem with per-interval edge travel times and τ_max |
| `problems/mttdvrp/state_mttdvrp.py` | new | Fleet state `s_F^t` + routing state `s_R^t`, transitions per §3.3 |
| `problems/__init__.py` | patch | Register `MTTDVRP` |
| `options.py` | patch | Add `mttdvrp` to the `--problem` help string |
| `tests/test_sed2am.py` | new | Smoke tests for the encoder + dual decoder |
| Everything else | unchanged from Kool 2019 |

## Datasets

The paper trains and evaluates on **real-world traffic data from Edmonton and Calgary, Canada** (§5.1). The synthetic instance generator in `problems/mttdvrp/problem_mttdvrp.py` produces uniform-in-unit-square instances with a default 5-piece speed profile across an 8-hour shift; pre-loaded city traffic tensors plug in via the `distribution` argument.

## Baselines

Per §5.3 the paper compares against AM, GCN-NPEC, GAT-Edge, Residual E-GAT, Google OR-Tools, the Clarke-Wright savings algorithm, GA, ALNS+VND, and DP+GA. SED2AM outperforms all of them on the Edmonton and Calgary corpora and generalises to larger instance sizes than it was trained on.

## Run

```bash
python run.py --problem mttdvrp --graph_size 50 --baseline rollout --run_name sed2am-n50
```

## Citation

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
  booktitle={ICLR},
  year={2019}
}
```

<!-- status 2024-12-29: model + tests stable -->

<!-- pre-submission sweep 2025-02-13 -->

<!-- arxiv header touched 2025-02-16 -->

<!-- aligned with arxiv v1 2025-03-04 -->

<!-- reviewer 2 wording 2025-03-22 -->

<!-- abstract-eq tightened 2025-05-02 -->

<!-- camera-ready typo sweep 2025-05-06 -->

<!-- TKDD DOI 2025-05-09 -->
<!-- maint 2025-01-04 -->

<!-- maint 2025-01-07 -->

<!-- maint 2025-02-11 -->

<!-- maint 2025-02-14 -->
