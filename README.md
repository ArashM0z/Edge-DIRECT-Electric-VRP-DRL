# SP-DE — Multi-Depot VRP with Inter-Depot Routes via Multi-Agent DRL

> **Forked from [wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route)** (Kool et al., ICLR 2019). This repo implements **SP-DE** (Single Policy, Distributed Execution) introduced in our Canadian AI 2025 paper.

[![Paper](https://img.shields.io/badge/Canadian%20AI-2025-blue)](https://caiac.pubpub.org/pub/w0os5i18)
[![Forked from](https://img.shields.io/badge/forked%20from-wouterkool/attention--learn--to--route-lightgrey)](https://github.com/wouterkool/attention-learn-to-route)

## What SP-DE is

SP-DE solves the **Multi-Depot Vehicle Routing Problem with Inter-Depot Routes (MDVRP-IDR)** — vehicles start at one of K depots, serve customers, and may return to *any* depot mid-tour to refill capacity. This is genuinely multi-agent: one actor per depot, coordinated through a centralised critic, but executing in parallel at inference.

The CTDE (centralised training, decentralised execution) pattern is the multi-agent RL standard. The encoder is shared; each depot has its own actor head with its own attention parameters; the critic is centralised and computes joint values.

## What this fork adds on top of Kool 2019

| File | What |
|---|---|
| `problems/mdvrp_idr/problem_mdvrp_idr.py` | MDVRP-IDR problem definition with multi-depot inputs |
| `problems/mdvrp_idr/state_mdvrp_idr.py` | State tracking per-vehicle (depot, location, capacity) + customer-visited mask |
| `nets/spde/shared_encoder.py` | Shared Transformer encoder over (depot + customer) graph |
| `nets/spde/per_depot_actor.py` | Per-depot actor (one PointerDecoder per depot) |
| `nets/spde/centralised_critic.py` | Attention-based centralised critic over joint agent embeddings |
| `nets/spde/spde_model.py` | Full SP-DE agent |

## Run

```bash
python run.py --problem mdvrp_idr --graph_size 50 --n_depots 3 --baseline rollout --run_name spde-n50
```

## Citation

```bibtex
@inproceedings{mozhdehi2025spde,
  title={{SP$\spadesuit$DE}: Solving the Multi-Depot Vehicle Routing Problem with Inter-Depot Routes Using Multi-Agent Deep Reinforcement Learning},
  author={Mozhdehi, Arash and Mohammadizadeh, Mahdi and Kalantari, Saeid and Kim, Beom Sae and Wang, Xin},
  booktitle={Canadian Conference on Artificial Intelligence},
  year={2025}
}
@inproceedings{kool2019attention,
  title={Attention, Learn to Solve Routing Problems!},
  author={Kool, Wouter and van Hoof, Herke and Welling, Max},
  booktitle={ICLR},
  year={2019}
}
```
