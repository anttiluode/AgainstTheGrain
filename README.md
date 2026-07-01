# Against the Grain

EDIT: Added adder3.html and how it works html visualizer:



### Why running a learned sequence backward is a different, unstable, energy-pumping process — not the forward one in reverse

**PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.**

> Do not hype. Do not lie. Just show.

---

## Where this came from

I drove into Helsinki and back the unusual way — out by bike, home by bike, against the grain of the trip I usually take by train. In Keskuspuisto I took a few wrong turns on a route I know cold. Not because I had forgotten it. Because I was running it **backward**, and backward is not the same machine as forward run in reverse.

That is the whole repo. The wrong turns are not a memory failure. They are the signature of a forward-built circuit being asked to run against its own arrow — and paying for it in instability, in a confidently-wrong internal predictor, and in a gate that opens on the wrong beats.

Everything here was already implied by the parent line. This repo only puts the reverse direction in the dock and reads the numbers off.

---

## The one idea

A learned sequence lives in an operator `C_τ = S ⊕ A`:

```
S = (M + Mᵀ)/2   the symmetric half — the held, settled "where". Arrow-blind.
A = (M − Mᵀ)/2   the skew half — the directed "when". THIS is the arrow.
```

Reversing time **negates A and leaves S untouched**. So the reverse is not a reflection of the forward computation — it is the *same settling* with the *arrow flipped*, and flipping the arrow is expensive three separate ways, each measured here in its own runnable file:

1. **the dynamics** — forward contracts error, backward amplifies it (`01`);
2. **the predictor** — a forward-trained mirror is confidently wrong in reverse, worse than no mirror (`02`);
3. **the gate** — the accelerometer that opens the backward channel is causal and asymmetric, so it opens on the wrong beats when the input is reversed (`03`).

But the arrow is only baked in when the route is held as a **worn 1D sequence**. Hold the same place as a **grid coordinate manifold** — the entorhinal torus — and reversal becomes nearly free (`04`). The cost of going backward is not a fact about "backward"; it is a fact about which map you are in.

---

## What the code shows (measured, seeded, reproducible — numpy only)

### `00_origin/against_the_grain.py` — the seed
The file that started this, from the conversation that became the repo. A dendrite reading sequence order, and the **spike-rate accelerometer** from Park, Cohen et al. (*Nat Commun* 2025) reduced to a gate. Its honest split set the whole direction:
- **[K]** a passive cable cannot *generate* a direction preference — a sigmoid-NMDA bolted on gives selectivity gain ~0.9×, forward/reverse work ratio 1.00× (the V13 wall). The arrow needs the *excitable* element (V14's 9.8:1), not passive geometry.
- **[V]** the accelerometer fires on onset/acceleration and goes dark under sustained firing — the `fSSS…fff` motif, concentration **2.2×** (window vs late). Ember's surprise gate, in biophysics, on the retrograde channel.

### `reverse/01_the_asymmetry.py` — the arrow is the skew half, and reverse is unstable and uphill
A linear dynamics **learned** (least squares) from a damped-spiral trajectory, decomposed `M = S ⊕ A`.

| claim | number | status |
|---|---|---|
| the arrow is A: it flips sign under reversal while S survives | cos(S,S_rev) **+0.99**, cos(A,A_rev) **−0.99** | **[V]** |
| reverse recall is ill-conditioned (forward contracts, M⁻¹ expands) | forward error growth **0.37×**, backward **1355×** over 20 steps | **[V]** |
| backward is uphill: forward sheds energy, backward must pump it | shed **−0.13**, pump **+0.20** | **[V]** |

Reverse is not forward-in-reverse. It is a **different, unstable, energy-pumping** process.

### `reverse/02_the_mirror_lies.py` — a forward-trained mirror poisons the reverse
A linear mirror learns the forward transition of a **smooth** trajectory (a route is locally smooth), then is run forward and backward.

| direction | mirror vs raw residual | reading |
|---|---|---|
| **forward** (predictable) | cancels **40%**, helps **1.67×** | a true mirror — passes only surprise |
| **backward** (all steps) | mirror/raw **1.51×** | **poison** — actively worse than no mirror |

A **2.5× swing** from mirror-as-help to mirror-as-hazard, purely by reversing the direction. This is the mirror-gate's law — *a confidently-wrong predictor is worse than none* — reproduced in the direction domain. The wrong turns are the forward world-model still firing, still confident, now pointing the wrong way.

### `reverse/03_the_gate_misfires.py` — the gate opens on the wrong beats backward
The accelerometer, driven by one acceleration and its exact time-reversal (same spikes, reflected).

| gate | accel vs reverse | reading |
|---|---|---|
| **level-keyed** (opens on rate *level*) | 12 vs 12, **1.0×** | **[K]** direction-blind — both start from silence; you cannot read the arrow off the level (the gate's V13 wall) |
| **lead-time-keyed** (Park's 15 ms prior depolarisation) | same count 7 vs 8, but **38% of beats misaligned** | **[V]** not time-reversal-symmetric — accel fires **late**, its reverse fires **early**; same count, **wrong timing** |

A gate tuned to the forward route opens on the wrong beats backward: silent where the surprise now is, firing where the road used to turn. And the fix is the parent line's fix — the arrow must be *generated* by causal asymmetric kinetics (V14), not read off a level (V13).

### `reverse/04_the_substrate_decides.py` — the cost of reverse depends on which map you hold
The **same round trip**, driven through two representations, with a small nudge injected at the turnaround.

| substrate | step map | energy on return | closure error after a 0.001 nudge |
|---|---|---|---|
| **worn 1D sequence** (dissipative attractor) | cond **1.49** | pumps **+61** (uphill) | **8.8** — the nudge blows up |
| **grid coordinate manifold** (toroidal phase code) | cond **1.00** (unitary) | **flat**, pump ~0 | **2.5e-3** — the nudge stays a nudge |

**Reverse is ~3600× more error-prone in the worn sequence than in the grid**, for the identical journey. A torus has no preferred direction: advancing and retreating a grid phase are equally cheap and exactly invertible. A worn, dissipative route-attractor is cheap only in the direction it was worn. So "against the grain" is expensive in the *habitual* substrate and nearly free in the *coordinate* one — the wrong turns are the habitual substrate failing in reverse and forcing a fallback onto the grid, which can do reverse but only when deliberately driven, never on habit's autopilot.

---

## The two substrates — what the grid-cell sweep paper adds, and what it doesn't

Vollan, Gardner, Moser & Moser (*Nature* 2025) recorded entorhinal–hippocampal ensembles and found that in each theta cycle the decoded position **sweeps outward** from the animal into the surroundings, with sweep direction **alternating left/right** on successive cycles (~80% of triplets), extending into **never-visited, inaccessible** locations, and **persisting through REM sleep**. The alternation is captured by an agent that simply **minimises overlap** with the space it has recently covered. A separate parasubiculum "internal direction" population drives the sweeps through conjunctive grid cells, whose projection onto pure grid cells is **asymmetric** — a directed phase offset aligned to the preferred direction (the bump-translation of attractor models).

This is the empirical backbone for the distinction `04` makes. The grid manifold is a **flexible, omnidirectional coordinate system** — it can sweep any direction, including where the animal has never been. That is the reversal-free substrate. The worn route-replay is the reversal-costly one. The paper grounds that both exist; it does not itself assign them the roles "habit" and "deliberate navigation" — that mapping is the interpretation the wrong turns fit, and it is a **bet**, not a result.

**On the analysis Gemini offered** (holding it to the same ledger as everything else here):

- **"Koopman free-run = sweeps into never-visited space"** — the most defensible bridge. Both are an internal generative process extrapolating beyond the sensed data; the sweeps' persistence in REM is exactly a free-running latent. Call it a **[B]** that earns its keep.
- **"tangent projector = overlap-minimising rule"** — a real shared motif: both are repulsion/decorrelation that spreads samples to cover a space. But one orthogonalises latent write-vectors and the other tiles physical 2D space with a moving beam. Family resemblance, not identity. **[B]**.
- **"±ω conjugate eigenpairs = left/right alternation"** — this one is poetry, not mechanism. The paper's own favoured account of the alternation is **firing-rate adaptation** (Chu; Ji; Widloski & Foster), an overlap/fatigue dynamic — not a readout of the two rotation senses of a skew operator's eigenplane. A skew operator's ±ω is the two spin directions of *one* plane; the sweep alternation is a temporal switch between *two headings*. Suggestive resonance, but calling them "the same object" overclaims. **[K]** as an identity; at most a loose **[B]** as analogy.
- **"running the hardware backward fights the asymmetric phase offset → the wrong turns"** — half right, and `04` is the correction. The *incremental translation* (conjunctive→grid) is genuinely directional, so path-integrating a step in reverse does fight it. But the grid *representation* as a whole is omnidirectional and reversal-cheap — so the catastrophic-reverse cost lives in the **worn sequence** substrate, not the grid. You don't run the grid backward; you re-aim the internal-direction signal and sweep the other way, cheaply. What's expensive is the habit.

The honest one-line version: the paper confirms the brain has **both** substrates; it does not confirm that your skew operator *is* the sweep generator. The value it adds to this repo is the **second substrate** — the one where reverse is free — which is what `04` now measures.

---

## The picture, assembled

When you run your habitual route **forward**:
- the settled half `S` holds the "where"; the skew half `A` supplies the "when" cheaply;
- the mirror has learned this direction, cancels the predictable, passes only surprise;
- the gate opens on the real onsets; the dynamics contract error toward the remembered path.
The cost is near zero. You drive it without thinking.

When you run it **backward** (against the grain):
- `A` is negated — the dynamics that *contracted* error forward now *amplify* it (1355× over 20 steps);
- the forward mirror confidently predicts the forward-next when the truth is the forward-prev — actively worse than having no model at all (1.51×);
- the gate, keyed to forward timing, opens on the wrong beats (38% misaligned) — silent at the new surprises, firing where the old turns were;
- and every step is uphill: you must *pump in* the energy the forward path sheds for free.

None of this is forgetting. It is a forward machine being asked to run against its own arrow. The wrong turns are the honest output of that.

---

## Run it

```bash
pip install numpy
python 00_origin/against_the_grain.py       # the seed: dendrite order + accelerometer
python reverse/01_the_asymmetry.py          # arrow = skew half; reverse unstable + uphill
python reverse/02_the_mirror_lies.py        # forward mirror poisons the reverse (2.5x swing)
python reverse/03_the_gate_misfires.py      # the gate opens on the wrong beats backward
```

Each prints its own numbers and its own verdict, kills included. Nothing is hidden in a figure the print-out doesn't also state.

---

## The honest ledger

**Verified in code (seeded, reproducible):**
- the direction of time is carried by the **skew half A** — it flips sign under reversal (cos −0.99) while S is preserved (cos +0.99);
- reverse recall is **ill-conditioned** — forward contracts error (0.37×), backward amplifies it (1355×), once the dynamics have a realistic spread of decay rates;
- backward is **uphill** — forward sheds energy, backward must pump it in;
- a forward-trained mirror is a **true mirror forward** (cancels 40%, helps 1.67×) and **poison backward** (1.51× worse than no mirror) — a 2.5× swing;
- the accelerometer, keyed on Park's 15 ms prior-depolarisation, is **not time-reversal-symmetric** — same dSpike count, 38% of beats on different moments.

**Killed by the builds (kept, because they teach):**
- "a passive cable / a sigmoid on it generates direction" — **false** (seed EXP1, work ratio 1.00×; the V13 wall). The arrow needs the excitable element.
- "a *level*-keyed gate reads direction" — **false** (`03`A, 1.0×): acceleration is a signed quantity `d(rate)/dt`; you cannot see the sign in the level.
- (the honest near-null that shaped `02`) "any forward mirror poisons any reversed stream" — **false**: on a stream of *unrelated* states the mirror is merely neutral (0.97×). The poison is specific to **smooth** trajectories — the regime a predictor is actually for, and the regime a route lives in.

**Honest limits — read before believing any of it:**
- linear maps, one toy trajectory each, few seeds, **relative units**, chosen parameters. These show an *architecture* produces an effect; none is calibrated to Joules, nats, metres of wrong turn, or any benchmark;
- `01`'s instability depends on the dynamics having a spread of decay rates (anisotropic dissipation) — true of real lossy systems, assumed here;
- `02`'s poison is a linear-mirror, smooth-trajectory result; a nonlinear predictor might recover in reverse, untested;
- `03`'s lead-time gate is a reduced two-variable stand-in for a channel network; the 38% misalignment is a *shape* statement, not a measured dendrite;
- "mirror", "gate", "reserve" are scalar abstractions of rich biology.

**The bet (untouched, as everywhere in this line):** that any of this reversal cost is *felt* — that the effort of driving against the grain is an experience and not only a larger residual. This locates the mechanism in code that can fail. It does not touch the hard problem.

---

## Lineage

Built directly on the PerceptionLab line (`github.com/anttiluode`):
- **`C_τ = S ⊕ A`** and *the arrow is the skew half* — GeometricNeuron V9;
- *reversing the arrow costs the entropy* (Crooks) — GeometricNeuron V20, `arrow_cost_proof.py`;
- *direction must be generated by an excitable element, not passive geometry* — ResonantNeuron V13→V14;
- *a fast inhibitory mirror teaches and cancels, and a confidently-wrong one is poison* — TheMaturingGate, `the_mirror_gate`;
- *the spike-rate accelerometer* — Park, Wong-Campos, Cohen et al., *Nat Commun* **16**:1333 (2025), the biophysics of the backward (bAP/dSpike) channel;
- *dendrites read input order* — Branco & Häusser (2010).

The framing — that reverse is a distinct, gated, uphill process and that the wrong turns are its signature — and the direction are Antti Luode's; the three experiments, the seed, and this ledger were built with Claude (Opus 4.8). MIT.

*Forward, the arrow carries you and the gate opens where the road turns. Backward, the arrow is negated, the error grows, the mirror lies, and the gate opens a beat too late. You do not forget the route. You pay to run it against its grain. Do not hype. Do not lie. Just show.*
