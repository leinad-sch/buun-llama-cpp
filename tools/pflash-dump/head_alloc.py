#!/usr/bin/env python3
# head_alloc.py — does HEAD-granularity K allocation beat LAYER-granularity at
# matched global bytes? Extends the validated layer RD-frontier allocator
# (vbr_frontier3_alloc.py) by optionally subdividing each global layer's K into
# its 4 KV-heads (each = 2 FWHT-128 coord-blocks = cb [h*2,h*2+2)). Per-head K
# marginal is MODELED by splitting the measured layer-K marginal across heads by
# d_eff share (higher d_eff => more sensitive => protect). V stays layer-grained
# (d_eff is K-only). Discrete tiers make per-layer matched-byte head-redistribution
# impossible, so the lever is the GLOBAL greedy's new freedom to mix tiers WITHIN a
# layer; matched bytes hold at the global budget. Validate combined (marginals
# ~3x sub-additive => ranking seed only).
#   policies: layer_greedy | head_greedy | head_blind
#     head_blind = head_greedy's global tier multiset, placed IGNORING d_eff
#                  (isolates "d_eff placement" from "more granularity").
# Usage: head_alloc.py <deff_csv> <target_bpv> <policy> [marg_tsv ctx]
#   marg_tsv/ctx optional: load K/V layer marginals from the unified
#   (layer side tier ctx kld) TSV instead of the hardcoded ctx2048 defaults
#   (used to drive the ctx8192 head-validation arm).
import sys, csv
from collections import Counter

BPW = {"t3": 14.0 / 32.0, "t4": 66.0 / 128.0, "t8": 130.0 / 128.0, "fp16": 16.0}
TIERS = ("t3", "t4", "t8", "fp16")
GLOBAL_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
HEAD_COORDS = 256          # per KV-head (head_dim)
LAYER_COORDS = 1024        # n_embd_k_gqa = 4 heads * 256
NHEAD = 4
BPH = 2                    # FWHT-128 blocks per head

# --- measured ctx2048 layer marginals (doc P1=t4, P2=t8, P5=t3), KLD vs fp16 ---
K_KLD = {
	"t4": [0.012717,0.010830,0.011672,0.008670,0.010932,0.008011,0.006240,0.006760,0.003698,0.001205,0.000661,0.000477,0.000369,0.000268,0.000258,0.000224],
	"t8": [0.007214,0.008013,0.006473,0.005043,0.004158,0.003712,0.002392,0.001584,0.000452,0.000297,0.000218,0.000189,0.000153,0.000123,0.000088,0.000034],
	"t3": [0.018011,0.014822,0.018431,0.016121,0.021291,0.011578,0.015062,0.013007,0.005784,0.002061,0.001840,0.001256,0.000958,0.000627,0.000688,0.000701],
}
V_KLD = {
	"t4": [0.011435,0.007464,0.008319,0.007253,0.005879,0.004996,0.003695,0.003316,0.000915,0.000577,0.000487,0.000316,0.000257,0.000215,0.000212,0.000816],
	"t8": [0.007059,0.005606,0.006594,0.004439,0.003231,0.004092,0.001202,0.001066,0.000481,0.000291,0.000219,0.000190,0.000149,0.000120,0.000087,0.000050],
	"t3": [0.014701,0.013143,0.018139,0.011465,0.011188,0.010135,0.009855,0.007644,0.002841,0.000938,0.000865,0.000714,0.000512,0.000441,0.000478,0.002919],
}

def load_marginals(path, ctx):
	# override K_KLD/V_KLD globals from the unified (layer side tier ctx kld) TSV
	tmp = {("k", t): [0.0] * len(GLOBAL_LAYERS) for t in ("t4", "t8", "t3")}
	tmp.update({("v", t): [0.0] * len(GLOBAL_LAYERS) for t in ("t4", "t8", "t3")})
	for line in open(path):
		p = line.strip().split("\t")
		if len(p) < 5 or p[0] == "layer":
			continue
		L, side, tier, c, mk = p
		if c != str(ctx) or mk == "-" or (side, tier) not in tmp:
			continue
		if int(L) in GLOBAL_LAYERS:
			tmp[(side, tier)][GLOBAL_LAYERS.index(int(L))] = float(mk)
	for t in ("t4", "t8", "t3"):
		K_KLD[t] = tmp[("k", t)]
		V_KLD[t] = tmp[("v", t)]

def load_deff(path):
	deff = {}
	for r in csv.DictReader(open(path)):
		de = float(r["d_eff"])
		if de <= 0.0:
			continue
		l = int(r["layer"]); h = int(r["kv_head"])
		deff.setdefault(l, {})[h] = de
	return deff

def kld_of(unit, tier, deff):
	# unit = (L,'v') | (L,'k') layer-gran | (L,'k',h) head-gran
	if tier == "fp16":
		return 0.0
	li = GLOBAL_LAYERS.index(unit[0])
	if unit[1] == "v":
		return V_KLD[tier][li]
	if len(unit) == 2:           # layer-gran K
		return K_KLD[tier][li]
	h = unit[2]                  # head-gran K
	dl = deff[unit[0]]
	share = dl[h] / sum(dl.values())
	return K_KLD[tier][li] * share

def coords(unit):
	return HEAD_COORDS if (unit[1] == "k" and len(unit) == 3) else LAYER_COORDS

def units_for(gran):
	u = []
	for L in GLOBAL_LAYERS:
		if gran == "head":
			for h in range(NHEAD):
				u.append((L, "k", h))
		else:
			u.append((L, "k"))
		u.append((L, "v"))
	return u

def hull(unit, deff):
	curve = [(t, BPW[t], kld_of(unit, t, deff)) for t in TIERS]
	pts = sorted(curve, key=lambda c: c[1])
	kept, best = [], float("inf")
	for t, b, k in pts:
		if k < best - 1e-12:
			kept.append((t, b, k)); best = k
	changed = True
	while changed and len(kept) >= 3:
		changed = False
		for i in range(1, len(kept) - 1):
			t0, b0, k0 = kept[i - 1]; t1, b1, k1 = kept[i]; t2, b2, k2 = kept[i + 1]
			if b2 == b0:
				continue
			interp = k0 + (k2 - k0) * (b1 - b0) / (b2 - b0)
			if k1 >= interp - 1e-12:
				kept.pop(i); changed = True; break
	return kept

def greedy(units, deff, target_bpv):
	hulls = {u: hull(u, deff) for u in units}
	tot_coords = sum(coords(u) for u in units)
	target_bytes = target_bpv * tot_coords
	def choose(lmb):
		amap, tb = {}, 0.0
		for u, h in hulls.items():
			wc = coords(u); best = None
			for t, b, k in h:
				j = k + lmb * b * wc
				if best is None or j < best[0]:
					best = (j, t, b)
			amap[u] = best[1]; tb += best[2] * wc
		return amap, tb
	lo, hi, best = 0.0, 1e6, None
	for _ in range(300):
		mid = 0.5 * (lo + hi)
		amap, tb = choose(mid)
		if best is None or abs(tb - target_bytes) < abs(best[1] - target_bytes):
			best = (amap, tb)
		if tb > target_bytes:
			lo = mid
		else:
			hi = mid
	return best[0]

def blind_place(head_greedy_map):
	# same per-layer K tier multiset as head_greedy, assigned to heads by head
	# index (0..3) paired with tiers sorted desc bpw -> placement blind to d_eff.
	out = {}
	for L in GLOBAL_LAYERS:
		ktiers = sorted([head_greedy_map[(L, "k", h)] for h in range(NHEAD)],
		                key=lambda t: -BPW[t])
		for h in range(NHEAD):
			out[(L, "k", h)] = ktiers[h]
		out[(L, "v")] = head_greedy_map[(L, "v")]
	return out

def emit(amap):
	parts = ["default=fp16"]
	for u in sorted(amap, key=lambda x: (x[0], x[1], x[2] if len(x) > 2 else -1)):
		t = amap[u]
		if t == "fp16":
			continue
		L = u[0]
		if u[1] == "v":
			parts.append("band=0-2000000000:%s:L%d-%d:v" % (t, L, L))
		elif len(u) == 2:
			parts.append("band=0-2000000000:%s:L%d-%d:k" % (t, L, L))
		else:
			h = u[2]
			parts.append("band=0-2000000000:%s:L%d-%d:k:c%d-%d" % (t, L, L, h * BPH, h * BPH + BPH))
	return ";".join(parts)

def avg_bpv(amap):
	tc = sum(coords(u) for u in amap)
	return sum(BPW[amap[u]] * coords(u) for u in amap) / tc

def main():
	deff_csv, target, policy = sys.argv[1], float(sys.argv[2]), sys.argv[3]
	if len(sys.argv) >= 6:
		load_marginals(sys.argv[4], int(sys.argv[5]))
	deff = load_deff(deff_csv)
	if policy == "layer_greedy":
		amap = greedy(units_for("layer"), deff, target)
	elif policy == "head_greedy":
		amap = greedy(units_for("head"), deff, target)
	elif policy == "head_blind":
		amap = blind_place(greedy(units_for("head"), deff, target))
	else:
		raise SystemExit("bad policy " + policy)
	ms = dict(Counter(amap.values()))
	sys.stderr.write("# target=%.4f policy=%-12s bytes/val=%.4f multiset=%s\n" % (
		target, policy, avg_bpv(amap), ms))
	print(emit(amap))

if __name__ == "__main__":
	main()
