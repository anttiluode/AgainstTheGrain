"""
02_the_mirror_lies.py
====================================================================
A forward-trained predictor cancels the forward stream and is
CONFIDENTLY WRONG on the reversed one -- worse than no predictor at all.

This is the mirror-gate's poison result (TheMaturingGate/the_mirror_gate:
a confident non-mirror teacher drags the student below learning-alone),
reproduced in the DIRECTION domain. The forward mirror is a true mirror
of forward time and a false one of reverse time.

World: a cyclic tour of K distinct patterns with rare surprise jumps
(the mirror-gate's world). A linear mirror W learns the forward
transition x_{t+1} ~ W x_t. We then measure the residual the downstream
principal cell must still encode -- forward vs backward, mirror vs raw.

  residual (mirror)  =  next_actual - W @ current
  residual (raw)     =  next_actual - current        (no predictor)

Forward: mirror should CANCEL (residual << raw)   -- it is a true mirror.
Backward: mirror should POISON (residual >  raw)   -- confidently wrong.

numpy only.
"""
import numpy as np

rng = np.random.default_rng(2)
d = 16
# A SMOOTH closed trajectory (a route is locally smooth: consecutive places
# resemble each other). Random low-harmonic loop in d-dim, closed. This is the
# regime a predictor is actually FOR -- and the regime the anecdote lives in
# (a car on a road), not a set of unrelated teleports.
n_harm = 3
amps = rng.standard_normal((n_harm, d)) / np.arange(1, n_harm + 1)[:, None]
phase = rng.uniform(0, 2*np.pi, (n_harm, d))
def curve(phi):
    x = np.zeros_like(phi)[..., None] * np.zeros(d)
    out = np.zeros((*np.shape(phi), d)) if np.ndim(phi) else np.zeros(d)
    for k in range(n_harm):
        out += np.cos((k+1) * np.atleast_1d(phi)[:, None] + phase[k]) * amps[k]
    return out if np.ndim(phi) else out[0]

# build a long stream: advance smoothly around the loop, with rare surprise
# teleports to a random phase.
T = 6000
surprise_rate = 0.10
dphi = 2*np.pi / 60.0            # ~60 steps per loop -> smooth, small increments
phis = np.zeros(T)
is_surprise = np.zeros(T - 1, dtype=bool)
for t in range(1, T):
    if rng.random() < surprise_rate:
        phis[t] = rng.uniform(0, 2*np.pi)      # surprise teleport
        is_surprise[t-1] = True
    else:
        phis[t] = (phis[t-1] + dphi) % (2*np.pi)
stream = curve(phis)                            # (T, d)
stream /= np.linalg.norm(stream, axis=1, keepdims=True).mean()   # scale ~O(1)

# --------------------------------------------------------------------
# fit the forward mirror W:  stream[t+1] ~ W stream[t]
# --------------------------------------------------------------------
Xf = stream[:-1]; Yf = stream[1:]
W, *_ = np.linalg.lstsq(Xf, Yf, rcond=None)
W = W.T                                        # so Y ~ W X

def residuals(direction):
    """return (mirror_resid, raw_resid) arrays over the stream."""
    if direction == "forward":
        cur, nxt = stream[:-1], stream[1:]
    else:  # backward: play the same stream in reverse; 'next' is the prior state
        cur, nxt = stream[1:], stream[:-1]
    mir = np.linalg.norm(nxt - (cur @ W.T), axis=1)
    raw = np.linalg.norm(nxt - cur,          axis=1)
    return mir, raw

mf, rf = residuals("forward")
mb, rb = residuals("backward")

# on FORWARD, split predictable vs surprise to show selective cancellation
pred_mask = ~is_surprise
print("=" * 68)
print("FORWARD  —  the mirror is a true mirror: it cancels the predictable")
print("=" * 68)
print(f"  predictable steps : mirror residual {mf[pred_mask].mean():.3f}"
      f"   raw {rf[pred_mask].mean():.3f}   cancelled "
      f"{100*(1 - mf[pred_mask].mean()/rf[pred_mask].mean()):.0f}%")
print(f"  surprise steps    : mirror residual {mf[~pred_mask].mean():.3f}"
      f"   raw {rf[~pred_mask].mean():.3f}")
print(f"  --> on the habitual route the mirror erases the input; the principal")
print(f"      cell only ever sees the surprise. (the tensor's spend-on-surprise)")
print()

print("=" * 68)
print("BACKWARD  —  the same mirror is confidently WRONG: it poisons")
print("=" * 68)
print(f"  all steps : mirror residual {mb.mean():.3f}   raw {rb.mean():.3f}")
poison = mb.mean() / rb.mean()
print(f"  mirror/raw ratio = {poison:.2f}   ( >1 means the mirror ACTIVELY HARMS )")
if poison > 1.05:
    verdict = "POISON: the forward mirror makes reverse WORSE than no mirror."
elif poison < 0.95:
    verdict = "still helps: mirror not directional enough in this world."
else:
    verdict = "neutral: mirror neither helps nor harms in reverse."
print(f"  --> {verdict}")
print()

print("=" * 68)
print("THE ASYMMETRY IN ONE NUMBER")
print("=" * 68)
fwd_gain = rf[pred_mask].mean() / mf[pred_mask].mean()   # how much it helps forward
bwd_gain = rb.mean() / mb.mean()                          # >1 helps, <1 harms backward
print(f"  forward  : mirror divides the residual by {fwd_gain:.2f}  (helps)")
print(f"  backward : mirror multiplies the residual by {1/bwd_gain:.2f}  ("
      f"{'harms' if bwd_gain<1 else 'helps'})")
print(f"  swing    : {fwd_gain*(1/bwd_gain if bwd_gain<1 else 1):.1f}x  from mirror-as-help")
print(f"             to mirror-as-hazard, purely by reversing the direction.")
print()

print("=" * 68)
print("LEDGER (02)")
print("=" * 68)
print(f"""  [V] the forward mirror cancels the forward predictable stream
      ({100*(1 - mf[pred_mask].mean()/rf[pred_mask].mean()):.0f}% cancelled) and passes only surprise -- a true mirror.
  [{'V' if poison>1.05 else 'K'}] on the reversed stream the SAME mirror is confidently wrong:
      mirror/raw ratio {poison:.2f} -- {'it actively harms' if poison>1.05 else 'it does not clearly harm'}, the mirror-gate
      poison result in the direction domain.
  The wrong turns are not forgetting. The forward world-model is still
  firing, still confident, and now pointing the wrong way -- and a
  confident wrong predictor is worse than none (the mirror-gate's law).""")
