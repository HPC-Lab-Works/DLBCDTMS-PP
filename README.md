# DLBCDTMS-PP: simulator, smart contracts and experiment scripts

Reproduces every number, table and figure in the revised manuscript.

| File | Purpose |
|---|---|
| `mobility.py` | Urban grid (Manhattan), highway and random-waypoint mobility |
| `schemes.py` | Proposed trust model and baseline abstractions (DLBTM, BDTM, BBC, LTM, BIBM) |
| `simulator.py` | Discrete-time simulator (events, reports, credibility, ratings, RSU coverage, blocks, attacks) |
| `calibrate.py` | Parameter and threshold calibration on seeds 1000-1009 |
| `run_experiments.py` | All experiments, 30 seeds each (seeds 0-29) -> `results/runs.jsonl` |
| `analyze.py` | Statistics (mean, SD, 95% CI, Wilcoxon + Holm), figures and LaTeX tables |
| `crypto_bench.py` | ECDSA, ECIES, AES-GCM, SHA-256 and proof-of-work timings |
| `contracts/DLBC.sol`, `gas_bench.py` | Registry, FASC and TESC contracts; gas measurement on py-evm |
| `consensus.py` | PBFT emulation; single-layer vs double-layer ledger costs |
| `privacy.py` | Pseudonym-linking adversary and pseudonym-change strategies |
| `overhead_tables.py` | Overhead, gas, consensus and privacy tables |

Order: `crypto_bench.py`, `gas_bench.py` (needs `solcjs` 0.8.24), `calibrate.py`, `run_experiments.py`, `analyze.py`, `consensus.py`, `privacy.py`, `overhead_tables.py`.

Requirements: Python 3.11+, numpy, scipy, matplotlib, cryptography, web3, eth-tester[py-evm]; Node.js with `npm i -g solc@0.8.24`.
