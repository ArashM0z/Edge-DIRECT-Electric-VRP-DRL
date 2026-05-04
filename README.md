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

<!-- m 2025-08-22T16:32:00-06:00 -->

<!-- m 2026-01-21T11:30:00-06:00 -->

<!-- m 2023-09-23T23:40:00-06:00 -->

<!-- m 2025-04-07T17:14:00-06:00 -->

<!-- m 2025-03-13T23:57:00-06:00 -->

<!-- m 2025-06-17T21:58:00-06:00 -->

<!-- m 2023-05-24T23:39:00-06:00 -->

<!-- m 2023-12-21T23:08:00-06:00 -->

<!-- m 2023-08-08T17:49:00-06:00 -->

<!-- m 2026-05-12T19:01:00-06:00 -->

<!-- m 2024-03-20T19:50:00-06:00 -->

<!-- m 2023-01-22T15:38:00-06:00 -->

<!-- m 2023-03-07T22:15:00-06:00 -->

<!-- m 2023-09-26T22:32:00-06:00 -->

<!-- m 2024-04-19T22:50:00-06:00 -->

<!-- m 2023-04-15T18:32:00-06:00 -->

<!-- m 2025-08-09T20:47:00-06:00 -->

<!-- m 2026-04-13T22:07:00-06:00 -->

<!-- m 2025-06-15T16:20:00-06:00 -->

<!-- m 2023-01-22T19:22:00-06:00 -->

<!-- m 2025-06-10T14:58:00-06:00 -->

<!-- m 2025-04-04T22:21:00-06:00 -->

<!-- m 2023-08-26T17:42:00-06:00 -->

<!-- m 2025-04-03T13:21:00-06:00 -->

<!-- m 2023-09-24T16:29:00-06:00 -->

<!-- m 2025-05-07T18:26:00-06:00 -->

<!-- m 2026-01-12T17:48:00-06:00 -->

<!-- m 2023-01-23T19:00:00-06:00 -->

<!-- m 2025-01-11T14:23:00-06:00 -->

<!-- m 2025-09-08T23:33:00-06:00 -->

<!-- m 2026-01-23T18:27:00-06:00 -->

<!-- m 2026-04-15T21:28:00-06:00 -->

<!-- m 2024-06-15T17:10:00-06:00 -->

<!-- m 2025-08-01T21:37:00-06:00 -->

<!-- m 2025-06-13T18:47:00-06:00 -->

<!-- m 2025-02-24T23:58:00-06:00 -->

<!-- m 2026-01-22T19:31:00-06:00 -->

<!-- m 2025-05-06T23:46:00-06:00 -->

<!-- m 2026-04-01T22:40:00-06:00 -->

<!-- m 2023-05-22T16:28:00-06:00 -->

<!-- m 2025-10-31T23:48:00-06:00 -->

<!-- m 2025-05-03T19:42:00-06:00 -->

<!-- m 2024-06-16T19:04:00-06:00 -->

<!-- m 2026-01-15T20:10:00-06:00 -->

<!-- m 2023-05-04T21:04:00-06:00 -->

<!-- m 2023-12-20T16:41:00-06:00 -->

<!-- m 2025-06-15T22:19:00-06:00 -->

<!-- m 2023-01-23T22:56:00-06:00 -->

<!-- m 2023-01-20T19:00:00-06:00 -->

<!-- m 2023-05-05T16:06:00-06:00 -->

<!-- m 2025-05-07T21:29:00-06:00 -->

<!-- m 2024-06-14T18:54:00-06:00 -->

<!-- m 2025-06-17T23:32:00-06:00 -->

<!-- m 2024-06-14T20:23:00-06:00 -->

<!-- m 2024-09-11T15:33:00-06:00 -->

<!-- m 2023-12-25T15:46:00-06:00 -->

<!-- m 2023-01-26T16:32:00-06:00 -->

<!-- m 2024-09-10T17:42:00-06:00 -->

<!-- m 2025-02-26T18:30:00-06:00 -->

<!-- m 2026-01-20T17:42:00-06:00 -->

<!-- m 2023-05-04T21:23:00-06:00 -->

<!-- m 2026-01-25T16:24:00-06:00 -->

<!-- m 2023-04-16T17:10:00-06:00 -->

<!-- m 2023-05-03T23:35:00-06:00 -->

<!-- m 2024-09-09T17:21:00-06:00 -->

<!-- m 2023-02-10T19:15:00-06:00 -->

<!-- m 2023-01-19T19:03:00-06:00 -->

<!-- m 2023-02-12T18:28:00-06:00 -->

<!-- m 2023-09-26T18:27:00-06:00 -->

<!-- m 2023-04-14T13:14:00-06:00 -->

<!-- m 2023-03-09T23:52:00-06:00 -->

<!-- m 2023-06-12T18:54:00-06:00 -->

<!-- m 2026-04-15T16:56:00-06:00 -->

<!-- m 2023-12-20T19:52:00-06:00 -->

<!-- m 2025-05-04T20:44:00-06:00 -->

<!-- m 2024-09-11T15:45:00-06:00 -->

<!-- m 2024-09-11T22:27:00-06:00 -->

<!-- m 2023-09-23T14:05:00-06:00 -->

<!-- m 2025-10-19T14:26:00-06:00 -->

<!-- m 2023-03-15T18:20:00-06:00 -->

<!-- m 2023-09-05T14:24:00-06:00 -->

<!-- m 2023-01-19T23:21:00-06:00 -->

<!-- m 2023-01-16T17:14:00-06:00 -->

<!-- m 2023-09-01T22:54:00-06:00 -->

<!-- m 2023-09-30T17:51:00-06:00 -->

<!-- m 2025-10-31T23:15:00-06:00 -->

<!-- m 2025-03-10T14:33:00-06:00 -->

<!-- m 2026-01-15T19:12:00-06:00 -->

<!-- m 2025-08-12T23:54:00-06:00 -->

<!-- m 2026-01-15T15:47:00-06:00 -->

<!-- m 2023-09-27T21:54:00-06:00 -->

<!-- m 2024-03-26T14:10:00-06:00 -->

<!-- m 2025-10-16T17:07:00-06:00 -->

<!-- m 2023-05-05T17:27:00-06:00 -->

<!-- m 2023-04-15T13:01:00-06:00 -->

<!-- m 2024-06-17T22:49:00-06:00 -->

<!-- m 2024-07-13T21:26:00-06:00 -->

<!-- m 2024-01-25T19:06:00-06:00 -->

<!-- m 2026-01-25T14:46:00-06:00 -->

<!-- m 2023-09-23T23:00:00-06:00 -->

<!-- m 2026-04-15T23:19:00-06:00 -->

<!-- m 2025-12-11T18:12:00-06:00 -->

<!-- m 2025-08-01T17:35:00-06:00 -->

<!-- m 2023-03-08T17:59:00-06:00 -->

<!-- m 2026-03-07T17:20:00-06:00 -->

<!-- m 2023-08-27T21:03:00-06:00 -->

<!-- m 2025-02-03T16:50:00-06:00 -->

<!-- m 2025-04-16T15:34:00-06:00 -->

<!-- m 2025-05-05T22:58:00-06:00 -->

<!-- m 2026-01-13T15:14:00-06:00 -->

<!-- m 2024-07-16T17:25:00-06:00 -->

<!-- m 2025-03-06T15:52:00-06:00 -->

<!-- m 2024-10-11T18:32:00-06:00 -->

<!-- m 2026-01-24T17:57:00-06:00 -->

<!-- m 2026-04-16T18:32:00-06:00 -->

<!-- m 2023-03-10T20:47:00-06:00 -->

<!-- m 2025-04-20T22:56:00-06:00 -->

<!-- m 2023-04-16T20:22:00-06:00 -->

<!-- m 2025-01-11T23:07:00-06:00 -->

<!-- m 2026-01-13T14:22:00-06:00 -->

<!-- m 2023-05-05T15:40:00-06:00 -->

<!-- m 2023-12-26T20:03:00-06:00 -->

<!-- m 2025-02-28T18:08:00-06:00 -->

<!-- m 2023-06-27T23:39:00-06:00 -->

<!-- m 2023-09-01T13:54:00-06:00 -->

<!-- m 2023-01-22T15:39:00-06:00 -->

<!-- m 2024-09-09T21:31:00-06:00 -->

<!-- m 2023-04-15T23:25:00-06:00 -->

<!-- m 2023-08-30T16:47:00-06:00 -->

<!-- m 2025-04-06T19:33:00-06:00 -->

<!-- m 2026-03-28T18:59:00-06:00 -->

<!-- m 2025-03-01T18:42:00-06:00 -->

<!-- m 2023-01-22T17:48:00-06:00 -->

<!-- m 2025-04-04T13:59:00-06:00 -->

<!-- m 2024-03-25T22:46:00-06:00 -->

<!-- m 2026-01-19T15:48:00-06:00 -->

<!-- m 2026-01-15T14:30:00-06:00 -->

<!-- m 2023-03-07T13:31:00-06:00 -->

<!-- m 2025-10-24T23:34:00-06:00 -->

<!-- m 2024-10-07T23:59:00-06:00 -->

<!-- m 2025-07-31T18:27:00-06:00 -->

<!-- m 2025-05-06T20:36:00-06:00 -->

<!-- m 2026-01-11T20:50:00-06:00 -->

<!-- m 2025-01-30T16:03:00-06:00 -->

<!-- m 2023-03-10T23:29:00-06:00 -->

<!-- m 2023-05-11T23:48:00-06:00 -->

<!-- m 2025-07-30T19:26:00-06:00 -->

<!-- m 2026-03-02T20:03:00-06:00 -->

<!-- m 2023-09-27T23:01:00-06:00 -->

<!-- m 2026-01-15T18:47:00-06:00 -->

<!-- m 2026-03-22T20:44:00-06:00 -->

<!-- m 2023-04-02T18:53:00-06:00 -->

<!-- m 2023-04-12T22:26:00-06:00 -->

<!-- burst 2024-07-30 #1 -->

<!-- burst 2024-07-30 #2 -->

<!-- burst 2024-07-30 #3 -->

<!-- burst 2024-07-30 #4 -->

<!-- burst 2024-07-30 #5 -->

<!-- burst 2024-07-30 #6 -->

<!-- burst 2024-07-30 #7 -->

<!-- burst 2024-07-30 #8 -->

<!-- burst 2024-07-30 #9 -->

<!-- burst 2024-07-30 #10 -->

<!-- burst 2024-07-30 #11 -->

<!-- burst 2024-07-30 #12 -->

<!-- burst 2024-07-30 #13 -->

<!-- burst 2024-07-30 #14 -->

<!-- burst 2024-07-30 #15 -->

<!-- burst 2024-07-30 #16 -->

<!-- burst 2024-07-30 #17 -->

<!-- burst 2024-07-30 #18 -->

<!-- burst 2024-07-30 #19 -->

<!-- burst 2024-07-30 #20 -->

<!-- burst 2024-07-30 #21 -->

<!-- burst 2024-07-30 #22 -->

<!-- burst 2024-08-01 #1 -->

<!-- burst 2024-08-01 #2 -->

<!-- burst 2024-08-01 #3 -->

<!-- burst 2024-08-01 #4 -->

<!-- burst 2024-08-01 #5 -->

<!-- burst 2024-08-01 #6 -->

<!-- burst 2024-08-01 #7 -->

<!-- burst 2024-08-01 #8 -->

<!-- burst 2024-08-01 #9 -->

<!-- burst 2024-08-01 #10 -->

<!-- burst 2024-08-01 #11 -->

<!-- burst 2024-08-01 #12 -->

<!-- burst 2024-08-01 #13 -->

<!-- burst 2024-08-01 #14 -->

<!-- burst 2024-08-01 #15 -->

<!-- burst 2024-08-01 #16 -->

<!-- burst 2024-08-01 #17 -->

<!-- burst 2024-08-01 #18 -->

<!-- burst 2024-08-01 #19 -->

<!-- burst 2024-08-01 #20 -->

<!-- burst 2024-08-01 #21 -->

<!-- burst 2024-08-01 #22 -->

<!-- burst 2024-08-01 #23 -->

<!-- burst 2024-08-01 #24 -->

<!-- burst 2026-05-04 #1 -->

<!-- burst 2026-05-04 #2 -->
