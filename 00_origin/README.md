# 00_origin — the seed

`against_the_grain.py` is the file that started this repo, built live in the
conversation that became it. Two experiments:

- **EXP 1 [K]** — a dendrite reading sequence order. A passive cable, and a
  sigmoid-NMDA bolted on, do NOT generate a direction preference (gain ~0.9x,
  forward/reverse work ratio 1.00x — the V13 wall). Reproducing real dendritic
  order-selectivity (Branco & Hausser 2010) needs the *excitable* element from
  ResonantNeuron V14 (FitzHugh-Nagumo, 9.8:1), which this file deliberately
  omits — so the null points cleanly at what the real build already has.

- **EXP 2 [V]** — the spike-rate accelerometer from Park, Cohen et al.
  (*Nat Commun* 2025), reduced to a gate: opened by A-type Kv inactivation,
  closed by slow Nav inactivation. It fires on onset/acceleration (the
  `fSSS...fff` motif) and is dark under sustained firing — concentration 2.2x.
  This IS Ember's surprise gate, in biophysics, and it gates the *backward*
  (retrograde bAP/dSpike) channel — the one the whole repo is about.

Run: `python against_the_grain.py`
