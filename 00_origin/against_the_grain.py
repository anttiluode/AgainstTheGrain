"""
against_the_grain.py
====================================================================
The reverse is not the mirror.

A minimal, self-contained demonstration of two claims that stand
behind Antti's Keskuspuisto wrong-turns anecdote, both grounded in
Park, Cohen et al. (Nat Commun 2025), "Dendritic excitations govern
back-propagation via a spike-rate accelerometer."

    EXP 1  DIRECTION SELECTIVITY
           A dendrite reads the ORDER of a spatiotemporal input.
           - passive lossy cable  -> nearly order-blind   (the V13 wall)
           - + NMDA-like boost     -> strongly order-selective (V14 fix)
           This is Branco & Hausser (2010) re-cast in the V13->V14 language:
           you cannot draw the arrow into passive geometry; an active,
           supralinear element has to generate it.

    EXP 2  THE ACCELEROMETER GATE  (= Ember's surprise gate, in biophysics)
           A reduced 2-channel model of the dSpike window:
              opened by A-type Kv inactivation,
              closed by slow Nav inactivation.
           It fires the BACKWARD signal (dSpike) on an ONSET/acceleration
           and goes dark during sustained firing -> a novelty/surprise gate
           on the retrograde channel. Reports a concentration number, the
           same currency Ember uses.

No hype. Just the numbers the print-out states.  numpy only.
"""
import numpy as np

rng = np.random.default_rng(0)

# ====================================================================
# EXP 1 — a dendrite reads sequence order
# ====================================================================
# Reduced multi-compartment cable. Compartment 0 = soma, N-1 = distal tip.
# A "route" activates the compartments in some spatial order, in time.
#   centripetal (IN)  : distal -> proximal, the habitual / forward sweep
#   centrifugal (OUT) : proximal -> distal, the same places, reversed
# Readout = peak somatic depolarisation.

N      = 8
dt     = 0.05          # ms
T      = 60.0          # ms
steps  = int(T / dt)
C_m    = 1.0
g_leak = 0.06
g_ax   = 0.9           # axial coupling between neighbours
E_leak = 0.0
E_syn  = 70.0          # mV above rest
tau_r, tau_d = 0.5, 4.0   # synaptic alpha rise/decay (ms)

def alpha(t):
    t = np.maximum(t, 0.0)
    return (np.exp(-t / tau_d) - np.exp(-t / tau_r))

def run_cable(onset_ms, nmda=False, g_ampa=0.06, g_nmda=0.22):
    """Simulate the cable; return peak soma Vm and total active work.

    AMPA is fast and weak (drives a small travelling wave). NMDA is SLOW
    (tau ~50 ms) and voltage-gated (graded Mg-block relief), so a synapse's
    NMDA conductance is still open when a later synapse fires -- and unblocks
    most where the membrane is already depolarised. A centripetal (IN) sweep
    builds that depolarisation ahead of each arriving input; a centrifugal
    (OUT) sweep does not. Kept subthreshold so nothing saturates to E_syn.
    """
    V = np.zeros(N)
    s_nmda = np.zeros(N)          # slow NMDA gating state per compartment
    tau_nmda = 50.0
    peak = 0.0
    work = 0.0
    tvec = np.arange(steps) * dt
    for i in range(steps):
        t = tvec[i]
        gsyn = g_ampa * alpha(t - onset_ms)
        gsyn = np.clip(gsyn, 0.0, None)
        I_ampa = gsyn * (E_syn - V)

        # NMDA: gate opens with input, decays slowly; current is Mg-gated
        s_nmda += dt * (gsyn - s_nmda / tau_nmda)
        I_nmda = np.zeros(N)
        if nmda:
            mg = 1.0 / (1.0 + np.exp(-(V - 20.0) / 6.0))    # graded Mg relief
            gn = g_nmda * s_nmda * mg
            I_nmda = gn * (E_syn - V)

        I_ax = np.zeros(N)
        I_ax[:-1] += g_ax * (V[1:] - V[:-1])
        I_ax[1:]  += g_ax * (V[:-1] - V[1:])

        dV = (-g_leak * (V - E_leak) + I_ampa + I_nmda + I_ax) / C_m
        V = V + dt * dV
        peak = max(peak, V[0])
        work += np.sum(np.abs(I_nmda)) * dt
    return peak, work

# onset schedule: a sweep of width `span` ms across the N compartments
span = 12.0
comp_delay = np.linspace(0, span, N)          # distal tip lags most
IN_onset  = comp_delay[::-1].copy()           # distal first  -> soma-directed
OUT_onset = comp_delay.copy()                 # proximal first -> outward

def dsi(peak_in, peak_out):
    return (peak_in - peak_out) / (peak_in + peak_out + 1e-12)

print("=" * 68)
print("EXP 1  —  a dendrite reads sequence order (Branco-Hausser via V13->V14)")
print("=" * 68)

p_in_pas,  _ = run_cable(IN_onset,  nmda=False)
p_out_pas, _ = run_cable(OUT_onset, nmda=False)
p_in_act,  w_in  = run_cable(IN_onset,  nmda=True)
p_out_act, w_out = run_cable(OUT_onset, nmda=True)

print(f"passive cable   : IN peak {p_in_pas:6.2f}   OUT peak {p_out_pas:6.2f}"
      f"   DSI {dsi(p_in_pas,p_out_pas):+.3f}")
print(f"+ NMDA boost    : IN peak {p_in_act:6.2f}   OUT peak {p_out_act:6.2f}"
      f"   DSI {dsi(p_in_act,p_out_act):+.3f}")
print()
print(f"active work spent : forward(IN) {w_in:6.2f}   reverse(OUT) {w_out:6.2f}"
      f"   ratio {w_in/(w_out+1e-9):.2f}x")
sel_gain = abs(dsi(p_in_act,p_out_act)) / (abs(dsi(p_in_pas,p_out_pas)) + 1e-9)
print(f"selectivity gain from the active element : {sel_gain:.1f}x")
print()
print("  Read [KILL]: in THIS reduced single cable the passive DSI (+0.20) is")
print("  just attenuation asymmetry -- proximal inputs reach the soma bigger --")
print("  not a true arrow, and the bolted-on NMDA sigmoid does NOT amplify it")
print("  (gain ~0.9x; forward/reverse active work 1.00x, a mini-echo of V13's")
print("  1.0000 wall). Reproducing real dendritic order-selectivity (Branco-")
print("  Hausser) needs the EXCITABLE element from V14 (FitzHugh-Nagumo, 9.8:1),")
print("  not a sigmoid on a passive cable. The toy re-teaches the parent lesson.")
print()

# ====================================================================
# EXP 2 — the spike-rate accelerometer = a surprise gate on the bAP
# ====================================================================
# Reduced Park et al. mechanism. Two slow state variables gate whether a
# back-propagating somatic spike (bAP) becomes a dendritic spike (dSpike):
#     kA_inact  in [0,1] : A-type Kv INACTIVATION. 0 = shunt on (window shut),
#                          1 = shunt inactivated (window OPEN). Depolarising
#                          activity inactivates it (opens window) over ~10-25 ms;
#                          it recovers (re-shuts) when quiet.
#     na_reserve in [0,1]: Nav slow-inactivation reserve. 1 = full, drains under
#                          sustained spiking (~100-180 ms), recovers when quiet.
# Rule (their Fig 4 / channel-reserve section):
#     dSpike  iff  window OPEN (kA_inact high)  AND  reserve available (na high)
#
# Drive: silence, then a burst (an ONSET / acceleration), then sustained firing.
# Expect the accelerometer motif: fail, success, success, ..., then fail.

def accelerometer(spike_times, T=400.0, dt=0.5):
    tau_open   = 10.0    # kA inactivation timeconstant (window opens)
    tau_reshut = 45.0    # kA recovery when quiet (window re-shuts)
    consume    = 0.085   # fraction of Nav reserve inactivated PER spike
    tau_recov  = 260.0   # Nav reserve recovery when quiet
    thr_win    = 0.45    # window-open threshold
    thr_res    = 0.40    # reserve threshold

    steps = int(T / dt)
    kA = 0.0             # start shut (shunt active)
    na = 1.0             # start full reserve
    spk = set(int(round(t / dt)) for t in spike_times)
    out = []             # (t, bAP, dSpike)
    recent = 0.0         # leaky spike-rate estimate
    for i in range(steps):
        t = i * dt
        spiking = i in spk
        recent = recent * np.exp(-dt / 20.0) + (1.0 if spiking else 0.0)
        # kA inactivation: opens with depolarising drive, re-shuts when quiet
        drive = 1.0 if recent > 0.15 else 0.0
        kA += dt * (drive * (1 - kA) / tau_open - (1 - drive) * kA / tau_reshut)
        # Nav reserve: discrete per-spike inactivation, slow recovery when quiet
        na += dt * (1 - na) / tau_recov
        if spiking:
            na *= (1.0 - consume)
        kA = np.clip(kA, 0, 1); na = np.clip(na, 0, 1)
        if spiking:
            dspike = (kA > thr_win) and (na > thr_res)
            out.append((t, kA, na, dspike))
    return out

print("=" * 68)
print("EXP 2  —  the accelerometer gate = a surprise gate on the backward signal")
print("=" * 68)

# a STEP stimulus (paper's Fig 4a protocol): sustained firing from a standing
# start. Expect fail (window shut) -> success (window opened by kA inactivation)
# -> ... -> fail (reserve drained by slow Nav inactivation). The accelerometer.
spikes = list(range(50, 380, 14))   # ~70 Hz step from t=50 ms
res = accelerometer(spikes)

print(" t(ms)   window(kA)  reserve(Na)   bAP -> dSpike?")
motif = []
for (t, kA, na, ds) in res:
    tag = "  dSPIKE" if ds else "  (fail)"
    motif.append("S" if ds else "f")
    if t < 200 or ds:   # print the interesting early window + any late successes
        print(f"{t:6.1f}     {kA:5.2f}       {na:5.2f}      spike{tag}")

# concentration: dSpike rate in the transient WINDOW (just after onset, window
# open + reserve full) vs during LATE sustained firing (reserve drained)
early = [ds for (t, kA, na, ds) in res if 60 <= t <= 170]
late  = [ds for (t, kA, na, ds) in res if t >= 260]
r_early = np.mean(early) if early else 0.0
r_late  = np.mean(late)  if late  else 0.0
conc = r_early / (r_late + 1e-9)
print()
print(f"  motif                : {''.join(motif)}   (f=fail, S=dSpike)")
print(f"  dSpike rate in window: {r_early:.2f}")
print(f"  dSpike rate sustained: {r_late:.2f}")
print(f"  concentration        : {conc:.1f}x  (window / late-sustained)")
print()
print("  Read: the backward signal is DARK when quiet, fires on the ONSET")
print("  (silence -> burst = an acceleration = a surprise), and goes dark again")
print("  under sustained firing. Same shape, same currency, as Ember's gate —")
print("  but here it is the retrograde channel, and it is real biophysics.")
print()
print("=" * 68)
print("THE LEDGER")
print("=" * 68)
print("""  [K] EXP1 in this toy: a passive cable's order-signal is attenuation, not
      an arrow, and a sigmoid-NMDA bolted on does not generate selectivity
      (gain 0.9x, work ratio 1.00x). Re-teaches V13: passive geometry is
      reciprocal. The arrow needs the V14 excitable element (9.8:1), which
      this file deliberately does NOT include -- so this is an honest null,
      pointing at what the real build already has.
  [V] EXP2: the backward channel is gated by an accelerometer that fires on
      onset/acceleration (fSSS...fff motif) and is dark during sustained
      firing (concentration ~2x, window vs late). This IS the surprise gate,
      same shape as Ember's, grounded in Park et al. 2025 -- and it gates the
      RETROGRADE signal, the one the anecdote says is unnatural and costly.
  [B] that this is WHY reversing a learned route (drive the unusual way,
      wrong turns in the park) costs more and errs more: consistent with the
      accelerometer + the V20 Crooks reverse-cost leg, but 'costs more to run
      backward' as a trained-recall claim still needs arrow_cost_proof.py
      wired in and an actual reverse-recall task. Not shown here. A bet.
""")
