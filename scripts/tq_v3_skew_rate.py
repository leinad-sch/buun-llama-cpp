#!/usr/bin/env python3
# Does the odd-moment (skew) prize grow at lower bitrates? Same faithful codec sim
# (unit-norm -> signed FWHT -> N-level book -> exact-L2 norm correction), swept over
# book sizes 2/4/8/16 levels (1/2/3/4 bits). Arms per rate:
#   stock  = Lloyd-Max for the Gaussian null at that rate (analytic, like deployed books)
#   sym    = symmetric Lloyd retrained on pooled real rotated coords
#   asym   = per-side skew-sign-aligned + asymmetric Lloyd (the odd-moment codec)
import numpy as np, sys, glob, os

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

def rot_frame(raw, mu):
	g = raw.reshape(-1, 8, D) - mu.reshape(8, D)
	gn = np.linalg.norm(g, axis=2, keepdims=True)
	return (g / np.where(gn > 1e-10, gn, 1.0)) @ ROT, gn

def lloyd(x, n_levels, symmetric, iters=120):
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

def gauss_lloyd(n_levels, sigma):
	# Lloyd-Max for N(0, sigma^2) via large Gaussian sample (matches deployed analytic books)
	rng = np.random.default_rng(11)
	return lloyd(rng.normal(0, sigma, 2_000_000), n_levels, symmetric=True)

def codec_mse(y, cent, flips, x_raw, mu, gn):
	ya = y * flips
	mid = 0.5 * (cent[:-1] + cent[1:])
	q = cent[np.searchsorted(mid, ya)] * flips
	rn = np.linalg.norm(q, axis=2, keepdims=True)
	cn = np.where(rn > 1e-10, gn / rn, gn)
	rec = (q @ ROT.T) * cn + mu.reshape(8, D)
	return float(((rec - x_raw.reshape(-1, 8, D)) ** 2).mean())

def main(dump_dir):
	# load everything once
	data = {}
	for side in ("k", "v"):
		for path in sorted(glob.glob(os.path.join(dump_dir, f"{side}_l*_w1024.f32"))):
			layer = int(os.path.basename(path).split("_l")[1][:3])
			raw = np.fromfile(path, dtype=np.float32).reshape(-1, 1024).astype(np.float64)
			mu = raw[: raw.shape[0] // 2].mean(axis=0)
			data[(side, layer)] = (raw, mu)
	# per-side global skew signs + pooled train samples
	rng = np.random.default_rng(3)
	gs, pool_plain, pool_al = {}, {"k": [], "v": []}, {"k": [], "v": []}
	for side in ("k", "v"):
		sks = []
		for (s, l), (raw, mu) in data.items():
			if s != side: continue
			ytr, _ = rot_frame(raw[: raw.shape[0] // 2], mu)
			m2 = (ytr ** 2).mean(axis=0)
			sks.append((ytr ** 3).mean(axis=0) / m2 ** 1.5)
		g = np.sign(np.median(np.stack(sks), axis=(0, 1))); g[g == 0] = 1
		gs[side] = g
		for (s, l), (raw, mu) in data.items():
			if s != side: continue
			ytr, _ = rot_frame(raw[: raw.shape[0] // 2], mu)
			sel = rng.choice(ytr.shape[0], 512, replace=False)
			pool_plain[side].append(ytr[sel].ravel())
			pool_al[side].append((ytr[sel] * g).ravel())
	sigma = INV   # unit rows: per-coord std ~ 1/sqrt(128), same normalization as deployed books
	print(f"{'bits':>4} {'side':>4} {'stockMSE':>9} {'sym/stock':>9} {'asym/stock':>10} {'asym/sym':>9}")
	for n_levels, bits in ((2, 1), (4, 2), (8, 3), (16, 4)):
		cb_g = gauss_lloyd(n_levels, sigma)
		for side in ("k", "v"):
			cb_s = lloyd(np.concatenate(pool_plain[side]), n_levels, symmetric=True)
			cb_a = lloyd(np.concatenate(pool_al[side]), n_levels, symmetric=False)
			r_s, r_a = [], []
			for (s, l), (raw, mu) in sorted(data.items()):
				if s != side: continue
				ev = raw[raw.shape[0] // 2 :]
				yev, gn = rot_frame(ev, mu)
				ones = np.ones((1, 1, D))
				base = codec_mse(yev, cb_g, ones, ev, mu, gn)
				r_s.append(codec_mse(yev, cb_s, ones, ev, mu, gn) / base)
				r_a.append(codec_mse(yev, cb_a, gs[side][None, None, :], ev, mu, gn) / base)
			gm = lambda a: float(np.exp(np.mean(np.log(a))))
			print(f"{bits:>4} {side:>4} {base:9.6f} {gm(r_s):9.4f} {gm(r_a):10.4f} {gm(r_a)/gm(r_s):9.4f}", flush=True)

if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else "/root/cert_dump_27b")
