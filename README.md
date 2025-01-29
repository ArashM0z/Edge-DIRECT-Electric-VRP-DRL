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

<!-- maint 2025-03-22 -->

<!-- maint 2025-03-26 -->

<!-- maint 2025-04-30 -->

<!-- maint 2025-05-03 -->

<!-- maint 2025-06-08 -->

<!-- maint 2025-06-12 -->

<!-- maint 2025-07-18 -->

<!-- maint 2025-07-20 -->

<!-- maint 2025-08-25 -->

<!-- maint 2025-08-28 -->

<!-- maint 2025-10-04 -->

<!-- maint 2025-10-07 -->

<!-- maint 2025-11-11 -->

<!-- maint 2025-11-14 -->

<!-- maint 2025-12-21 -->

<!-- maint 2025-12-24 -->

<!-- maint 2024-01-05 -->

<!-- maint 2024-01-08 -->

<!-- maint 2024-02-25 -->

<!-- maint 2024-02-29 -->

<!-- maint 2024-04-17 -->

<!-- maint 2024-04-22 -->

<!-- maint 2024-06-09 -->

<!-- maint 2024-06-13 -->

<!-- maint 2024-07-31 -->

<!-- maint 2024-08-03 -->

<!-- maint 2024-09-20 -->

<!-- maint 2024-09-24 -->

<!-- maint 2024-11-11 -->

<!-- maint 2024-11-16 -->

<!-- maint 2023-01-09 -->

<!-- maint 2023-01-15 -->

<!-- maint 2023-03-15 -->

<!-- maint 2023-03-21 -->

<!-- iter 2023-02-13-09 -->

<!-- iter 2023-02-13-11 -->

<!-- iter 2023-02-13-13 -->

<!-- iter 2023-02-13-15 -->

<!-- iter 2023-02-13-17 -->

<!-- iter 2023-02-13-19 -->

<!-- iter 2023-02-20-09 -->

<!-- iter 2023-02-20-11 -->

<!-- iter 2023-02-20-13 -->

<!-- iter 2023-02-20-15 -->

<!-- iter 2023-02-20-17 -->

<!-- iter 2023-02-20-19 -->

<!-- iter 2023-02-20-21 -->

<!-- iter 2023-07-10-09 -->

<!-- iter 2023-07-10-11 -->

<!-- iter 2023-07-10-13 -->

<!-- iter 2023-07-10-15 -->

<!-- iter 2023-07-10-17 -->

<!-- iter 2023-07-10-19 -->

<!-- iter 2023-07-24-09 -->

<!-- iter 2023-07-24-11 -->

<!-- iter 2023-07-24-13 -->

<!-- iter 2023-07-24-15 -->

<!-- iter 2023-07-24-17 -->

<!-- iter 2023-07-24-19 -->

<!-- iter 2023-07-24-21 -->

<!-- iter 2023-12-04-09 -->

<!-- iter 2023-12-04-11 -->

<!-- iter 2023-12-04-13 -->

<!-- iter 2023-12-04-15 -->

<!-- iter 2023-12-04-17 -->

<!-- iter 2023-12-04-19 -->

<!-- iter 2023-12-18-09 -->

<!-- iter 2023-12-18-11 -->

<!-- iter 2023-12-18-13 -->

<!-- iter 2023-12-18-15 -->

<!-- iter 2023-12-18-17 -->

<!-- iter 2023-12-18-19 -->

<!-- iter 2023-12-18-21 -->

<!-- iter 2023-12-18-22 -->

<!-- iter 2024-01-08-09 -->

<!-- iter 2024-01-08-11 -->

<!-- iter 2024-01-08-13 -->

<!-- iter 2024-01-08-15 -->

<!-- iter 2024-01-08-17 -->

<!-- iter 2024-01-08-19 -->

<!-- iter 2024-01-08-21 -->

<!-- iter 2024-01-15-09 -->

<!-- iter 2024-01-15-11 -->

<!-- iter 2024-01-15-13 -->

<!-- iter 2024-01-15-15 -->

<!-- iter 2024-01-15-17 -->

<!-- iter 2024-01-15-19 -->

<!-- iter 2024-01-15-21 -->

<!-- iter 2024-01-15-22 -->

<!-- iter 2024-03-18-09 -->

<!-- iter 2024-03-18-11 -->

<!-- iter 2024-03-18-13 -->

<!-- iter 2024-03-18-15 -->

<!-- iter 2024-03-18-17 -->

<!-- iter 2024-03-18-19 -->

<!-- iter 2024-03-18-21 -->

<!-- iter 2024-03-18-22 -->

<!-- iter 2024-03-25-09 -->

<!-- iter 2024-03-25-11 -->

<!-- iter 2024-03-25-13 -->

<!-- iter 2024-03-25-15 -->

<!-- iter 2024-03-25-17 -->

<!-- iter 2024-03-25-19 -->

<!-- iter 2024-09-30-09 -->

<!-- iter 2024-09-30-11 -->

<!-- iter 2024-09-30-13 -->

<!-- iter 2024-09-30-15 -->

<!-- iter 2024-09-30-17 -->

<!-- iter 2024-09-30-19 -->

<!-- iter 2026-01-12-09 -->

<!-- iter 2026-01-12-11 -->

<!-- iter 2026-01-12-13 -->

<!-- iter 2026-01-12-15 -->

<!-- iter 2026-01-12-17 -->

<!-- iter 2026-01-12-19 -->

<!-- iter 2026-01-12-21 -->

<!-- iter 2026-01-19-09 -->

<!-- iter 2026-01-19-11 -->

<!-- iter 2026-01-19-13 -->

<!-- iter 2026-01-19-15 -->

<!-- iter 2026-01-19-17 -->

<!-- iter 2026-01-19-19 -->

<!-- iter 2026-01-19-21 -->

<!-- iter 2026-01-19-22 -->

<!-- m 2026-01-20T21:57:00-06:00 -->

<!-- m 2023-06-10T14:09:00-06:00 -->

<!-- m 2024-03-24T20:07:00-06:00 -->

<!-- m 2026-04-29T23:12:00-06:00 -->

<!-- m 2026-04-01T23:20:00-06:00 -->

<!-- m 2024-04-28T14:38:00-06:00 -->

<!-- m 2024-05-23T19:38:00-06:00 -->

<!-- m 2023-05-22T18:54:00-06:00 -->

<!-- m 2025-09-11T18:22:00-06:00 -->

<!-- m 2025-01-29T08:52:00-06:00 -->
