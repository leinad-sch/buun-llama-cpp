#!/usr/bin/env python3
# gen_random_draws.py — extra random-arm draws for the EXP-15e head floor.
# Emits sched_random_sN.txt for each seed in argv, matched-budget identical to
# gen_head_schedule.py's random arm (2 heads t8 + 2 heads t3 per global layer,
# K only, c-block = head*2..+2). Used to turn the n=1 permutation floor into n>=6.
#   usage: gen_random_draws.py CSV outdir seed1 seed2 ...
import csv, sys, random

BPH = 2
N_HI = 2
POS_HI = 100000

def load(path):
	by_layer = {}
	for r in csv.DictReader(open(path)):
		de = float(r["d_eff"])
		if de <= 0.0:
			continue
		l = int(r["layer"]); h = int(r["kv_head"])
		by_layer.setdefault(l, []).append(h)
	return by_layer

def band(layer, head, tier):
	lo = head * BPH; hi = lo + BPH
	return "band=0-%d:%d:L%d-%d:k:c%d-%d" % (POS_HI, tier, layer, layer, lo, hi)

def schedule(by_layer, rng):
	parts = ["default=16"]
	for l in sorted(by_layer):
		hs = by_layer[l]
		hi = set(rng.sample(hs, N_HI))
		for h in hs:
			parts.append(band(l, h, 8 if h in hi else 3))
	return ";".join(parts)

def main():
	path = sys.argv[1]
	outdir = sys.argv[2]
	seeds = [int(s) for s in sys.argv[3:]]
	by_layer = load(path)
	for s in seeds:
		rng = random.Random(s)
		sched = schedule(by_layer, rng)
		open("%s/sched_random_s%d.txt" % (outdir, s), "w").write(sched + "\n")
		sys.stderr.write("seed %d -> %d bands\n" % (s, sched.count("band=")))

if __name__ == "__main__":
	main()
