#!/usr/bin/env python3
# Generate vanilla-book sweep candidates (.f32 files for TURBO_CB_T2/T3/T4/T8) from real
# 27B rows. Families per type: pooled(K+V) / K-only / V-only Lloyd retrains, plus scale
# perturbations of the pooled book (the gauge argument says overall scale is a no-op, so
# perturb RATIOS: inner/outer split), plus t8 companded (Lloyd-256 on |x|<=absmax frame).
import numpy as np, struct, sys, glob, os

D = 128
INV = 1.0 / np.sqrt(D)
S1 = np.array([-1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
	-1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
	-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
	-1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1], dtype=np.float64)
S2 = np.array([1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
	1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
	1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
	1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1], dtype=np.float64)

def hadamard(n):
	H = np.array([[1.0]])
	while H.shape[0] < n:
		H = np.block([[H, H], [H, -H]])
	return H

ROT = (np.diag(S1) @ (hadamard(D) * INV) @ np.diag(S2)).T

def lloyd(x, n_levels, symmetric=True, iters=150):
	c = np.quantile(x, (np.arange(n_levels) + 0.5) / n_levels)
	for _ in range(iters):
		mid = 0.5 * (c[:-1] + c[1:])
		idx = np.searchsorted(mid, x)
		newc = np.array([x[idx == i].mean() if np.any(idx == i) else c[i] for i in range(n_levels)])
		if symmetric:
			newc = 0.5 * (newc - newc[::-1])
		if np.max(np.abs(newc - c)) < 1e-9:
			c = newc; break
		c = newc
	return np.sort(c)

def w(path, vals):
	open(path, "wb").write(struct.pack(f"{len(vals)}f", *np.asarray(vals, dtype=np.float64)))
	print(path, np.round(vals, 5) if len(vals) <= 16 else f"[{len(vals)} levels]")

def main(dump_dir, out_dir):
	os.makedirs(out_dir, exist_ok=True)
	rng = np.random.default_rng(3)
	pool = {"k": [], "v": []}          # unit-norm rotated coords
	pool_am = {"k": [], "v": []}       # absmax-normalized rotated coords (t8 frame)
	for side in ("k", "v"):
		for path in sorted(glob.glob(os.path.join(dump_dir, f"{side}_l*_w1024.f32"))):
			raw = np.fromfile(path, dtype=np.float32).reshape(-1, 1024).astype(np.float64)
			mu = raw[: raw.shape[0] // 2].mean(axis=0)
			g = raw.reshape(-1, 8, D) - mu.reshape(8, D)
			gn = np.linalg.norm(g, axis=2, keepdims=True)
			y = (g / np.where(gn > 1e-10, gn, 1.0)) @ ROT
			sel = rng.choice(y.shape[0], min(1024, y.shape[0]), replace=False)
			ys = y[sel]
			pool[side].append(ys.ravel())
			am = np.abs(ys).max(axis=2, keepdims=True)
			pool_am[side].append((ys / np.where(am > 1e-10, am, 1.0)).ravel())
	both = np.concatenate(pool["k"] + pool["v"])
	ks   = np.concatenate(pool["k"])
	vs   = np.concatenate(pool["v"])
	for n, tag in ((4, "t2"), (8, "t3"), (16, "t4")):
		cb_p = lloyd(both, n); w(f"{out_dir}/{tag}_pooled.f32", cb_p)
		w(f"{out_dir}/{tag}_konly.f32", lloyd(ks, n))
		w(f"{out_dir}/{tag}_vonly.f32", lloyd(vs, n))
		# ratio perturbations of the pooled book: stretch/compress OUTER level only (gauge
		# makes overall scale free, so this explores the un-gauged ratio axis directly)
		for f, name in ((0.95, "out95"), (1.05, "out105")):
			cb = cb_p.copy(); cb[0] *= f; cb[-1] *= f
			w(f"{out_dir}/{tag}_{name}.f32", np.sort(cb))
	# t8 companded: symmetric Lloyd-256 on the absmax frame (values within [-1,1])
	both_am = np.concatenate(pool_am["k"] + pool_am["v"])
	w(f"{out_dir}/t8_companded.f32", lloyd(both_am, 256))
	print("done")

if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else "/root/cert_dump_27b",
	     sys.argv[2] if len(sys.argv) > 2 else "/root/night/cbswap/books")
