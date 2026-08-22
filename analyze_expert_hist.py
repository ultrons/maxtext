"""Analyze recorded expert histograms: optimal assignment, GBS-scaling stability, drift.

Input: expert_hist/step_*.npz, each containing arrays stacking to [num_layers, num_experts]
(scan-stacked plus possibly separate unscanned layers). Outputs, per layer and aggregate:
  1. optimal pi via greedy bin-pack (LPT) of mean expert loads into n_ranks bins,
     with imbalance = max-rank-load / mean-rank-load, before vs after.
  2. GBS bootstrap: sum k in {1,2,4,8,16} consecutive-step histograms -> imbalance of the
     STEP-1-derived pi on the k-batch load (does a pi inferred early hold at larger GBS?).
  3. drift: correlation of per-expert mean loads, first vs last quartile of training.
"""
import numpy as np, glob, sys

d = sys.argv[1] if len(sys.argv) > 1 else "expert_hist"
files = sorted(glob.glob(f"{d}/step_*.npz"))
H = []
for f in files:
    z = np.load(f)
    arrs = [z[k] for k in z.files]
    h = np.concatenate([a.reshape(-1, a.shape[-1]) for a in arrs], axis=0)  # [layers, E]
    H.append(h)
H = np.stack(H)                       # [steps, layers, E]
# Drop dense/MTP slots: rows that record no tokens in any step are not MoE layers.
moe_rows = H.sum(axis=(0, 2)) > 0
H = H[:, moe_rows, :]
S, L, E = H.shape
R = 8                                  # EP ranks
print(f"steps={S} moe_layers={L} (of {moe_rows.size} slots) experts={E} ranks={R}")
print(f"per-layer tokensxtopk per step: {sorted(set(H.sum(-1)[0].astype(int)))}")

def lpt(load, R):
    order = np.argsort(-load)
    bins = np.zeros(R); asg = np.zeros(len(load), int)
    for e in order:
        b = np.argmin(bins); bins[b] += load[e]; asg[e] = b
    return asg, bins

def imb(load, asg, R):
    rl = np.bincount(asg, weights=load, minlength=R)
    return rl.max() / rl.mean()

group_asg = np.repeat(np.arange(R), E // R)   # today: contiguous groups = ranks

mean_load = H.mean(0)                          # [layers, E]
cur = np.array([imb(mean_load[l], group_asg, R) for l in range(L)])
opt = []
for l in range(L):
    asg, _ = lpt(mean_load[l], R)
    opt.append(imb(mean_load[l], asg, R))
opt = np.array(opt)
print(f"\n1) imbalance (max/mean rank load), mean over layers:")
print(f"   today (group=rank): {cur.mean():.3f}   (worst layer {cur.max():.3f})")
print(f"   optimal pi (LPT):   {opt.mean():.3f}   (worst layer {opt.max():.3f})")

print(f"\n2) does an EARLY pi hold at larger GBS? (pi from steps 0-3 mean; applied to k-step sums)")
pi_early = [lpt(H[:4].mean(0)[l], R)[0] for l in range(L)]
for k in [1, 2, 4, 8, 16]:
    if S < k: break
    chunks = H[: (S // k) * k].reshape(-1, k, L, E).sum(1)   # [S/k, layers, E]
    v = np.array([[imb(c[l], pi_early[l], R) for l in range(L)] for c in chunks])
    g = np.array([[imb(c[l], group_asg, R) for l in range(L)] for c in chunks])
    print(f"   k={k:2d}: early-pi imbalance {v.mean():.3f}+-{v.std():.3f}   group=rank {g.mean():.3f}")

q = max(S // 4, 1)
early, late = H[:q].mean(0), H[-q:].mean(0)
corr = np.array([np.corrcoef(early[l], late[l])[0, 1] for l in range(L)])
print(f"\n3) drift: corr(early-quartile load, late-quartile load) mean {np.nanmean(corr):.3f} min {np.nanmin(corr):.3f}")
