"""
01_the_asymmetry.py
====================================================================
The arrow lives in the skew half. Reversing negates it, and the reverse
traversal is ill-conditioned and must pump energy in.

We LEARN a linear dynamics M from a trajectory (least squares, not by
hand), decompose M = S (+) A, and ask three falsifiable questions:

  (1) Is the direction of time carried by A, not S?
      -> fit M on the trajectory, fit M_rev on the REVERSED trajectory.
         S should survive (S = S^T is arrow-blind); A should FLIP SIGN.

  (2) Does running backward amplify error where forward suppresses it?
      -> forward dynamics are dissipative (a damped spiral, singular
         values < 1). M contracts; M^{-1} expands. Inject noise, step
         forward k vs backward k, compare error growth.

  (3) Does backward cost energy that forward releases?
      -> each forward step sheds energy (spontaneous, downhill); each
         backward step must PUMP energy in against the dissipation.
         Sum the pumped energy backward vs the shed energy forward.

Grounded in the parent line: C_tau = S (+) A (GeometricNeuron V9), the
skew half is the arrow (the_rotation_half_grounded), reversing the arrow
costs the entropy (V20 arrow_cost_proof / Crooks). Here it is a learned
map and a recall task. numpy only.
"""
import numpy as np

rng = np.random.default_rng(1)
n = 6                      # state dimension

# --------------------------------------------------------------------
# a "learned" damped-spiral trajectory in n dimensions
# --------------------------------------------------------------------
# generator: rotation in a few planes (the arrow) + ANISOTROPIC damping.
# Real lossy dynamics have a SPREAD of decay rates (fast and slow modes),
# not one uniform contraction. That spread is exactly what makes M^-1
# ill-conditioned -- so we build it honestly and read the cond number off.
theta = 0.35
plane_damp = [0.97, 0.85, 0.65]          # slow, medium, fast-decaying planes
M_true = np.zeros((n, n))
for j, k in enumerate(range(0, n - 1, 2)):   # planes (0,1),(2,3),(4,5)
    d = plane_damp[j]
    c, s = np.cos(theta), np.sin(theta)
    M_true[k, k]   =  d * c; M_true[k, k+1]   = -d * s
    M_true[k+1, k] =  d * s; M_true[k+1, k+1] =  d * c

# roll a trajectory
T = 4000
X = np.zeros((T, n))
X[0] = rng.standard_normal(n)
for t in range(1, T):
    X[t] = M_true @ X[t-1] + 0.01 * rng.standard_normal(n)

def fit_map(traj):
    """least-squares one-step map: traj[t+1] ~ M traj[t]."""
    A_ = traj[:-1]; B_ = traj[1:]
    M, *_ = np.linalg.lstsq(A_, B_, rcond=None)   # solves A_ M = B_
    return M.T                                     # so that B ~ M A

M   = fit_map(X)                 # fitted forward map
Mr  = fit_map(X[::-1])           # fitted map of the REVERSED trajectory

def split(Mx):
    S = 0.5 * (Mx + Mx.T)
    A = 0.5 * (Mx - Mx.T)
    return S, A

S, A   = split(M)
Sr, Ar = split(Mr)

print("=" * 68)
print("(1)  the arrow is the skew half A")
print("=" * 68)
def frob(x): return np.linalg.norm(x)
cos_S = np.sum(S * Sr) / (frob(S) * frob(Sr) + 1e-12)
cos_A = np.sum(A * Ar) / (frob(A) * frob(Ar) + 1e-12)
print(f"  ||S|| {frob(S):.3f}   ||A|| {frob(A):.3f}")
print(f"  cos( S_forward , S_reversed ) = {cos_S:+.3f}   (arrow-blind: stays ~ +1)")
print(f"  cos( A_forward , A_reversed ) = {cos_A:+.3f}   (the arrow: flips to ~ -1)")
print(f"  --> reversing time negates A while S survives. The direction of")
print(f"      time is written in the skew half, exactly as C_tau = S (+) A says.")
print()

print("=" * 68)
print("(2)  backward AMPLIFIES error where forward SUPPRESSES it")
print("=" * 68)
sv = np.linalg.svd(M, compute_uv=False)
cond = sv[0] / sv[-1]
print(f"  singular values of M : {np.array2string(sv, precision=3)}")
print(f"  spectral radius |M|  : {max(abs(np.linalg.eigvals(M))):.3f}   (<1: forward contracts)")
print(f"  condition number     : {cond:.2f}   (M^-1 expands the smallest mode)")

Minv = np.linalg.inv(M)
k = 20
eps = 1e-3
x0 = X[T // 2]
# forward: propagate x0 and x0+noise, compare divergence
def divergence(step_matrix, steps):
    a = x0.copy(); b = x0 + eps * rng.standard_normal(n)
    d0 = np.linalg.norm(a - b)
    for _ in range(steps):
        a = step_matrix @ a; b = step_matrix @ b
    return np.linalg.norm(a - b) / (d0 + 1e-12)

amp_fwd = divergence(M,    k)
amp_bwd = divergence(Minv, k)
print(f"  error growth over {k} steps  forward : {amp_fwd:8.3f}x")
print(f"  error growth over {k} steps  backward: {amp_bwd:8.3f}x")
print(f"  --> backward is {amp_bwd/ (amp_fwd+1e-12):.1f}x more error-amplifying than forward.")
print(f"      Reverse recall is not the mirror of forward recall: it is unstable.")
print()

print("=" * 68)
print("(3)  backward must PUMP energy that forward RELEASES")
print("=" * 68)
# walk a clean forward trajectory, then walk it backward with M^-1
seg = X[T // 2 : T // 2 + 200]
E = lambda v: float(v @ v)
shed = 0.0      # forward: sum of (E_out - E_in), expect negative (downhill)
for t in range(len(seg) - 1):
    shed += E(M @ seg[t]) - E(seg[t])
pump = 0.0      # backward: sum of (E_out - E_in) using M^-1, expect positive (uphill)
for t in range(len(seg) - 1, 0, -1):
    pump += E(Minv @ seg[t]) - E(seg[t])
print(f"  forward energy shed  (spontaneous, downhill): {shed:+.3f}")
print(f"  backward energy pumped (must be supplied)    : {pump:+.3f}")
print(f"  --> forward runs downhill for free; backward has to be driven,")
print(f"      and the work supplied is the entropy of reversing the arrow.")
print()

print("=" * 68)
print("LEDGER (01)")
print("=" * 68)
print(f"""  [V] the direction of time is carried by the skew half A: it flips
      sign under reversal (cos {cos_A:+.2f}) while S is preserved (cos {cos_S:+.2f}).
  [V] reverse recall is ill-conditioned: {amp_bwd/(amp_fwd+1e-12):.1f}x more error growth than
      forward over {k} steps, because forward contracts and M^-1 expands.
  [V] backward is uphill: forward sheds energy ({shed:+.1f}); backward must
      pump it in ({pump:+.1f}). Reversing the arrow costs work.
  Together: reverse is a different, unstable, energy-pumping process --
  not forward played in reverse. This is the substrate of the wrong turns.""")
