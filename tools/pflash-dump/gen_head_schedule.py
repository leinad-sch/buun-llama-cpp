#!/usr/bin/env python3
# gen_head_schedule.py — EXP-15e per-head KV matched-budget permutation test.
# Reads the d_eff CSV (layer,kv_head,n_samples,total_var,d_eff,top_coord_frac)
# and emits three TURBO_RAGGED_SCHEDULE strings at IDENTICAL byte budget:
#   protect-high : t8 on the 2 highest-d_eff heads/layer, t3 on the 2 lowest
#   protect-low  : the reverse (t8 on lowest d_eff)
#   random       : t8 on 2 random heads/layer
# K only, global-attn layers only (captured = nonzero d_eff). V + SWA-K stay fp16.
# head h occupies coord-blocks [h*BPH, (h+1)*BPH) with BPH = d_head/128 = 2.
import csv, sys, random

BPH = 2          # 128-FWHT blocks per head (d_head 256 / 128)
N_HI = 2         # heads promoted to t8 per layer
POS_HI = 100000  # cover all positions
SEED = 1905

def load(path):
	rows = [r for r in csv.DictReader(open(path))]
	by_layer = {}
	for r in rows:
		de = float(r["d_eff"])
		if de <= 0.0:
			continue          # SWA / uncaptured layer
		l = int(r["layer"]); h = int(r["kv_head"])
		by_layer.setdefault(l, []).append((h, de))
	return by_layer

def band(layer, head, tier):
	lo = head * BPH; hi = lo + BPH
	return "band=0-%d:%d:L%d-%d:k:c%d-%d" % (POS_HI, tier, layer, layer, lo, hi)

def schedule(by_layer, pick_hi):
	parts = ["default=16"]
	for l in sorted(by_layer):
		heads = by_layer[l]
		hi_set = pick_hi(l, heads)
		for h, _ in heads:
			parts.append(band(l, h, 8 if h in hi_set else 3))
	return ";".join(parts)

def main():
	path = sys.argv[1] if len(sys.argv) > 1 else "exp15e_head_deff.csv"
	by_layer = load(path)
	rng = random.Random(SEED)

	def hi_by_deff(rev):
		def f(l, heads):
			order = sorted(heads, key=lambda x: x[1], reverse=rev)
			return set(h for h, _ in order[:N_HI])
		return f

	def hi_random(l, heads):
		hs = [h for h, _ in heads]
		return set(rng.sample(hs, N_HI))

	arms = {
		"protect_high": schedule(by_layer, hi_by_deff(True)),
		"protect_low":  schedule(by_layer, hi_by_deff(False)),
		"random":       schedule(by_layer, hi_random),
	}
	for name, sched in arms.items():
		open("sched_%s.txt" % name, "w").write(sched + "\n")
		nb = sched.count("band=")
		sys.stderr.write("%-12s %d bands -> sched_%s.txt\n" % (name, nb, name))
	# echo the per-layer high-d_eff picks for the log
	for l in sorted(by_layer):
		order = sorted(by_layer[l], key=lambda x: x[1], reverse=True)
		sys.stderr.write("L%-2d  hi=%s  lo=%s\n" % (
			l, [h for h, _ in order[:N_HI]], [h for h, _ in order[N_HI:]]))

if __name__ == "__main__":
	main()
