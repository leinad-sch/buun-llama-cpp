#!/usr/bin/env python3
"""Train 1-bit TCQ codebook (k=1, L=8, 256 states, free initial state).
Mirrors the CUDA k_set_rows_turbo1_tcq trellis: next_state = (s>>1)|(out<<7).
Vectorized Viterbi (numpy over all 256 states) so real-data training is fast.
Exports a 256-float32 .bin (scaled by 1/sqrt(128)) for the TURBO1_TCQ_CB runtime loader.

Data:
  --data FILE.bin   post-FWHT unit-norm KV groups (float32 [-1,128]) from TURBO_EXTRACT (/tmp/turbo_postrot.bin).
  --tags FILE.bin   parallel int32 tags (/tmp/turbo_postrot_tags.bin); --kvsel 0=K(tag%1000//100==0) 1=V(==1).
  (no --data)       synthetic N(0,1) (sanity).
"""
import numpy as np, argparse

LLOYD_MAX_1BIT = np.array([-0.7978845608, 0.7978845608])
L, K = 8, 1
N_STATES = 1 << L
# Precompute trellis predecessor structure (static).
_low7 = np.arange(N_STATES) & 0x7F
_PRED0 = (_low7 << 1)            # predecessor with shifted-out bit 0
_PRED1 = _PRED0 | 1             # predecessor with shifted-out bit 1

def init_codebook(sigma=1.0):
    n_groups = 1 << (L - K)
    c = LLOYD_MAX_1BIT * sigma
    shifts = np.linspace(-(c[1]-c[0])/2, (c[1]-c[0])/2, n_groups, endpoint=False)
    cb = np.zeros(N_STATES)
    for g in range(n_groups):
        for p in range(2):
            cb[(g << K) | p] = c[p] + shifts[g]
    return cb

def encode_batch(X, cb):
    """Vectorized Viterbi over a batch X [B,T]; returns states [B,T] (int) and recon [B,T]."""
    B, T = X.shape
    INF = 1e30
    cost = np.zeros((B, N_STATES))           # free initial state
    bt = np.empty((T, B, N_STATES), dtype=np.int16)
    for t in range(T):
        c0 = cost[:, _PRED0]; c1 = cost[:, _PRED1]
        choose1 = c1 < c0
        best = np.where(choose1, c1, c0)
        bt[t] = (_PRED0[None, :] | choose1.astype(np.int16))   # actual predecessor state
        d = X[:, t][:, None] - cb[None, :]
        cost = best + d * d
    states = np.empty((B, T), dtype=np.int32)
    s = np.argmin(cost, axis=1).astype(np.int32)
    bidx = np.arange(B)
    for t in range(T - 1, -1, -1):
        states[:, t] = s
        s = bt[t][bidx, s].astype(np.int32)
    recon = cb[states]
    return states, recon

def train(data, n_iters=30, batch=4096):
    cb = init_codebook()
    best_mse, best_cb, prev = float('inf'), cb.copy(), float('inf')
    N = len(data)
    print(f"Training 1-bit TCQ k={K} L={L} ({N_STATES} states) on {N} groups")
    for it in range(n_iters):
        sums = np.zeros(N_STATES); cnts = np.zeros(N_STATES, dtype=np.int64); sse = 0.0
        for s0 in range(0, N, batch):
            X = data[s0:s0+batch]
            st, recon = encode_batch(X, cb)
            sse += float(((X - recon) ** 2).sum())
            flat = st.ravel(); xf = X.ravel()
            sums += np.bincount(flat, weights=xf, minlength=N_STATES)
            cnts += np.bincount(flat, minlength=N_STATES)
        mse = sse / (N * data.shape[1])
        star = ""
        if mse < best_mse: best_mse, best_cb, star = mse, cb.copy(), " *"
        nz = cnts > 0
        cb = cb.copy(); cb[nz] = sums[nz] / cnts[nz]
        print(f"  iter {it+1:2d}: MSE={mse:.6f}{star}  ({int(nz.sum())}/{N_STATES} used)")
        if it > 5 and abs(prev - mse) / max(mse, 1e-9) < 1e-4:
            print("  converged"); break
        prev = mse
    return best_cb, best_mse

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data'); ap.add_argument('--tags'); ap.add_argument('--kvsel', type=int, default=-1)
    ap.add_argument('--out', default='/tmp/tcq1_k.bin'); ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--iters', type=int, default=30); ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    np.random.seed(a.seed)
    scale = 1.0 / np.sqrt(128)
    if a.data:
        d = np.fromfile(a.data, dtype=np.float32).reshape(-1, 128).astype(np.float64)
        if a.tags and a.kvsel >= 0:
            tg = np.fromfile(a.tags, dtype=np.int32)
            tg = tg[:len(d)]
            sel = ((tg % 1000) // 100) == a.kvsel    # tag = layer*1000 + kvsel*100 + group
            d = d[sel]
            print(f"tag-filter kvsel={a.kvsel} ({'K' if a.kvsel==0 else 'V'}): {len(d)} of {len(tg)} records")
        # unit-norm groups -> scale to per-coord std 1 (×sqrt(128)) so the std-1 codebook init +
        # the ×1/sqrt(128) export match the CUDA encoder (which quantizes unit-norm post-FWHT coords).
        d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12) * np.sqrt(128)
        if len(d) > a.n: d = d[np.random.choice(len(d), a.n, replace=False)]
        data = d
        print(f"training on {len(data)} real groups from {a.data}")
    else:
        data = np.random.randn(a.n, 128); print(f"synthetic {len(data)} N(0,1) groups")
    cb, mse = train(data, n_iters=a.iters)
    (cb * scale).astype(np.float32).tofile(a.out)
    print(f"\nbest MSE={mse:.6f}; wrote {a.out} (256 float32, scaled 1/sqrt128)")
