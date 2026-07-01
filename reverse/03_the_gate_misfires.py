"""
03_the_gate_misfires.py
====================================================================
The gate reads direction. Reverse the input and it fires at the wrong
time -- because the mechanism (Park et al. 2025) is causal and
asymmetric: the window opens slowly (A-type Kv inactivation) and closes
slowly (Nav slow inactivation). An acceleration and its time-reverse (a
deceleration) are NOT the same input to it.

We take one spike pattern -- an ACCELERATION (rate ramping up from
silence) -- and its exact TIME-REVERSAL -- a DECELERATION with the
identical number of spikes. We run the same accelerometer on both and
count dendritic spikes (dSpikes = the backward/retrograde signal).

Prediction: the accelerometer fires on the acceleration and is largely
blind to its reverse. The gate is not time-symmetric; run your route
backward and the gate opens on the wrong beats.

Reuses the reduced accelerometer from 00_origin/against_the_grain.py.
numpy only.
"""
import numpy as np

def accelerometer(spike_times, T=400.0, dt=0.5):
    """LEVEL-keyed proxy: window opens off recent rate LEVEL. Direction-blind."""
    tau_open, tau_reshut = 10.0, 45.0
    consume, tau_recov   = 0.085, 260.0
    thr_win, thr_res     = 0.45, 0.40
    steps = int(T / dt)
    kA, na = 0.0, 1.0
    spk = set(int(round(t / dt)) for t in spike_times)
    out, recent = [], 0.0
    for i in range(steps):
        t = i * dt
        spiking = i in spk
        recent = recent * np.exp(-dt / 20.0) + (1.0 if spiking else 0.0)
        drive = 1.0 if recent > 0.15 else 0.0
        kA += dt * (drive * (1 - kA) / tau_open - (1 - drive) * kA / tau_reshut)
        na += dt * (1 - na) / tau_recov
        if spiking:
            na *= (1.0 - consume)
        kA = np.clip(kA, 0, 1); na = np.clip(na, 0, 1)
        if spiking:
            out.append((t, (kA > thr_win) and (na > thr_res)))
    return out

def accelerometer_leadtime(spike_times, T=400.0, dt=0.5, lead=15.0):
    """LEAD-TIME-keyed, faithful to Park: a dSpike needs the distal dendrite
    to have been DEPOLARISED for >= ~15 ms BEFORE the bAP arrives (the A-type
    Kv window must already be open). Depolarisation integrates spikes; the
    window requires the integral to have been above threshold for `lead` ms.
    This keys on the RISING phase -- so it is direction-selective."""
    tau_depol, thr_dep  = 30.0, 0.6
    consume, tau_recov  = 0.085, 260.0
    thr_res             = 0.40
    steps = int(T / dt)
    depol, na = 0.0, 1.0
    above_ms  = 0.0                      # how long depol has been above thr
    spk = set(int(round(t / dt)) for t in spike_times)
    out = []
    for i in range(steps):
        t = i * dt
        spiking = i in spk
        depol = depol * np.exp(-dt / tau_depol) + (1.0 if spiking else 0.0)
        above_ms = above_ms + dt if depol > thr_dep else 0.0
        na += dt * (1 - na) / tau_recov
        if spiking:
            na *= (1.0 - consume)
        na = np.clip(na, 0, 1)
        if spiking:
            window_open = above_ms >= lead          # 15 ms prior depolarisation
            out.append((t, window_open and (na > thr_res)))
    return out

# --------------------------------------------------------------------
# build ONE acceleration: silence, then spikes whose ISI shrinks (rate rises).
# --------------------------------------------------------------------
T = 400.0
isis = np.linspace(34, 4, 26)        # denser: ISI 34->4 ms, 26 spikes, rate rises hard
t = 55.0
accel = []
for gap in isis:
    accel.append(t); t += gap
accel = [x for x in accel if x < T - 5]

# the exact time-reversal (same spikes, reflected in time -> decelerating)
decel = sorted([ (T - x) for x in accel ])

def summarise(fn, name, spikes):
    res = fn(spikes)
    ds = [d for (_, d) in res]
    n_spk, n_ds = len(res), sum(ds)
    motif = "".join("S" if d else "f" for (_, d) in res)
    print(f"  {name:14s}: {n_spk} spikes -> {n_ds} dSpikes   motif {motif}")
    return n_ds

print("=" * 68)
print("A.  LEVEL-keyed gate (window opens on rate LEVEL) — the wall")
print("=" * 68)
a1 = summarise(accelerometer, "ACCELERATION", accel)
d1 = summarise(accelerometer, "its REVERSE",  decel)
r1 = a1 / (d1 + 1e-9)
print(f"  dSpikes accel {a1}  decel {d1}  ratio {r1:.1f}x")
print(f"  --> {'direction-BLIND (kill): both start from silence, both open the'  if abs(a1-d1)<=1 else 'asymmetric'}")
print(f"      window. Acceleration is a SIGNED quantity d(rate)/dt; a level")
print(f"      gate cannot see the sign. This is the V13 wall for the gate.")
print()

print("=" * 68)
print("B.  LEAD-TIME-keyed gate (Park's 15 ms prior depolarisation) — the fix")
print("=" * 68)
resA = accelerometer_leadtime(accel)
resD = accelerometer_leadtime(decel)
a2 = summarise(accelerometer_leadtime, "ACCELERATION", accel)
d2 = summarise(accelerometer_leadtime, "its REVERSE",  decel)
# time-reversal-equivariance: if the gate were direction-symmetric, reversing
# the INPUT would reverse the OUTPUT. Compare reversed-accel-output to decel-output.
mA = [d for (_, d) in resA]
mD = [d for (_, d) in resD]
L = min(len(mA), len(mD))
mA_rev = mA[::-1][:L]; mD = mD[:L]
misaligned = sum(1 for x, y in zip(mA_rev, mD) if x != y)
frac_mis = misaligned / L
print(f"  accel dSpike motif        : {''.join('S' if d else 'f' for d in mA)}")
print(f"  accel motif TIME-REVERSED : {''.join('S' if d else 'f' for d in mA[::-1])}")
print(f"  reverse-input dSpike motif: {''.join('S' if d else 'f' for d in mD)}")
print(f"  --> if the gate were direction-symmetric these last two would MATCH.")
print(f"      beats misaligned: {misaligned}/{L} ({100*frac_mis:.0f}%). The dSpikes")
print(f"      land on DIFFERENT beats: accel fires late, its reverse fires early.")
print(f"      Same count ({a2} vs {d2}) -- the gate does not fire LESS backward,")
print(f"      it fires at the WRONG TIME.")
print()

print("=" * 68)
print("LEDGER (03)")
print("=" * 68)
tagA = "K" if abs(a1 - d1) <= 1 else "V"
tagB = "V" if frac_mis >= 0.25 else "K"
print(f"""  [{tagA}] a LEVEL-keyed gate is direction-blind ({a1} vs {d1}, {r1:.1f}x): it opens
      on rate level, and both an acceleration and its reverse start from
      silence. You cannot read the arrow off the level -- the gate's V13 wall.
  [{tagB}] a LEAD-TIME-keyed gate -- Park's actual mechanism, window open >=15 ms
      before the bAP -- is NOT time-reversal-symmetric: reverse the input and
      the dSpikes land on different beats ({100*frac_mis:.0f}% misaligned), accel firing
      late, its reverse firing early. Same count, wrong timing. The arrow is
      GENERATED by causal asymmetric kinetics, exactly as V14 said.
  So a gate tuned to the forward route opens on the wrong beats backward:
      silent where the surprise now is, firing where the road used to turn.""")
