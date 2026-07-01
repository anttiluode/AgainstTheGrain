"""
04_the_substrate_decides.py
====================================================================
The cost of running backward is not a property of "backward". It is a
property of WHICH MAP you hold the route in.

Antti's correction (and the correction to an over-eager reading of the
Vollan/Moser 2025 grid-sweep paper): the same location can be held two
ways, and they sit at OPPOSITE ends of the reversal-cost axis this repo
measures.

  A. the WORN 1D SEQUENCE  -- an over-learned, dissipative attractor that
     replays the habitual route. This is what 01-03 are about: the arrow
     is baked in (skew A / forward mirror / causal gate), so reversing is
     unstable, poisoned, mistimed. Cheap FORWARD (automatic), dear BACKWARD.

  B. the GRID COORDINATE MANIFOLD -- position as phases on a few toroidal
     modules (entorhinal grid code). Motion is a CONSERVATIVE, BIJECTIVE
     phase translation. A torus has no preferred direction: advancing and
     retreating a phase are equally cheap and EXACTLY invertible. Reversal
     is free -- but the substrate is not automatic; it must be actively
     driven (the parasubiculum internal-direction signal), i.e. deliberate.

The test: drive the SAME round trip in both substrates, inject a small
perturbation at the turnaround, and measure (i) closure error on return
and (ii) energy that had to be pumped in. Same journey, two substrates.

numpy only.
"""
import numpy as np

rng = np.random.default_rng(4)
k = 20            # steps out, then steps back
eps = 1e-3        # perturbation injected at the turnaround

# ====================================================================
# A. the worn 1D sequence: a dissipative, anisotropic learned attractor
#    (the same object as 01 -- forward contracts, backward is ill-conditioned)
# ====================================================================
n = 6
theta = 0.35
plane_damp = [0.97, 0.85, 0.65]
M = np.zeros((n, n))
for j, kk in enumerate(range(0, n - 1, 2)):
    d = plane_damp[j]; c, s = np.cos(theta), np.sin(theta)
    M[kk, kk] = d*c; M[kk, kk+1] = -d*s
    M[kk+1, kk] = d*s; M[kk+1, kk+1] = d*c
Minv = np.linalg.inv(M)
E = lambda v: float(np.vdot(v, v).real)

x0 = rng.standard_normal(n)
# out-and-back with a perturbation at the turnaround
x = x0.copy(); shed = 0.0
for _ in range(k):
    xn = M @ x; shed += E(xn) - E(x); x = xn
x = x + eps * rng.standard_normal(n)                 # perturb at turnaround
pump = 0.0
for _ in range(k):
    xn = Minv @ x; pump += E(xn) - E(x); x = xn
closure_A = np.linalg.norm(x - x0)

print("=" * 68)
print("A. the worn 1D sequence  (dissipative learned attractor)")
print("=" * 68)
print(f"  condition number of the step map : {np.linalg.cond(M):.2f}")
print(f"  energy shed going out  : {shed:+.3f}   (downhill, free)")
print(f"  energy pumped coming back: {pump:+.3f}   (uphill, must be supplied)")
print(f"  closure error on return : {closure_A:.3f}   (a {eps:g} nudge blows up)")
print()

# ====================================================================
# B. the grid coordinate manifold: position as phases on toroidal modules
#    motion = conservative, bijective phase translation (unitary)
# ====================================================================
spacings = np.array([0.50, 0.80, 1.20])          # 3 grid modules (paper's scales)
# a smooth 2D velocity sequence for the SAME round trip length
angles = np.cumsum(rng.normal(0, 0.3, k))
vel = 0.05 * np.c_[np.cos(angles), np.sin(angles)]   # (k,2) velocities out

# encode position as unit phasors per module, per spatial axis
def phases_of(v_step, s):
    # phase advance for a velocity step on a module of spacing s (x and y axes)
    return 2*np.pi * v_step / s

z = np.ones((len(spacings), 2), dtype=complex)       # start phase 0 on all modules/axes
z0 = z.copy()
energy_trace = [np.sum(np.abs(z)**2)]
for t in range(k):                                    # go out
    for m, s in enumerate(spacings):
        dphi = phases_of(vel[t], s)
        z[m] *= np.exp(1j * dphi)
    energy_trace.append(np.sum(np.abs(z)**2))
# perturb the phase at the turnaround
z = z * np.exp(1j * eps * rng.standard_normal(z.shape))
for t in range(k - 1, -1, -1):                        # come back: negate velocity
    for m, s in enumerate(spacings):
        dphi = phases_of(vel[t], s)
        z[m] *= np.exp(-1j * dphi)
    energy_trace.append(np.sum(np.abs(z)**2))
closure_B = np.linalg.norm(np.angle(z / z0))          # phase closure error
energy_trace = np.array(energy_trace)

print("=" * 68)
print("B. the grid coordinate manifold  (conservative phase translation)")
print("=" * 68)
print(f"  condition number of the step map : 1.00   (rigid rotation, unitary)")
print(f"  energy range over the whole trip : "
      f"{energy_trace.min():.3f} .. {energy_trace.max():.3f}   (flat -> pump 0)")
print(f"  closure error on return : {closure_B:.2e}   (the {eps:g} nudge stays a nudge)")
print()

print("=" * 68)
print("SAME JOURNEY, TWO SUBSTRATES")
print("=" * 68)
print(f"  closure error  A (worn sequence) {closure_A:8.3f}   "
      f"B (grid coord) {closure_B:.2e}")
print(f"  reverse is {closure_A/ (closure_B+1e-12):.0f}x more error-prone in the worn")
print(f"  sequence than in the grid coordinate manifold, for the identical trip.")
print()
print("  Read: 'against the grain' is expensive in the WORN 1D SEQUENCE -- the")
print("  automatic, over-learned route replay (01-03). It is nearly FREE in the")
print("  GRID COORDINATE MANIFOLD, which a torus makes direction-symmetric. The")
print("  wrong turns come from the habitual substrate failing in reverse and")
print("  forcing a fallback onto the grid -- which CAN do reverse, but only when")
print("  actively, deliberately driven, not on habit's autopilot.")
print()

print("=" * 68)
print("LEDGER (04)")
print("=" * 68)
print(f"""  [V] reversal cost is substrate-dependent, not intrinsic to 'backward':
      worn 1D sequence -> closure error {closure_A:.2f}, energy pumped {pump:+.2f};
      grid coordinate manifold -> closure error {closure_B:.1e}, energy pump ~0.
  [V] a torus (grid code) is time-reversal symmetric: the step map is a
      rigid rotation (cond 1), so advancing and retreating cost the same
      and invert exactly. The habitual attractor is dissipative (cond {np.linalg.cond(M):.1f})
      and only cheap in the direction it was worn.
  [B] the brain uses BOTH (Vollan/Moser 2025: a flexible grid manifold that
      sweeps omnidirectionally into never-visited space, plus directional
      bump-translation machinery). Mapping habit->sequence and deliberate
      navigation->grid is the interpretation the wrong turns fit; the paper
      grounds the two substrates but does not assign them these two roles.""")
