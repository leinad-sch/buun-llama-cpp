#!/usr/bin/env python3
# gen_masks.py — EXP-15d content axis as a PRICE SWEEP.
#
# Our allocator prices precision per block. Here the price of a block is
#     priority = z_position + weight * z_content
# where z_position is the model's average positional importance (recency-shaped,
# our "basin" baseline) and z_content is the per-window deviation above that baseline
# (what THIS window attends to beyond position = the content outliers). `weight` is
# the PRICE we put on content: weight=0 is the pure positional baseline; raising it
# lets an important token bid its way to a higher tier over position-protected blocks.
# We do not know the right weight, so we sweep it.
#
# Every emitted mask spends the SAME bytes: an identical graded tier histogram
# (n8*t8, n4*t4, n3*t3) over the 64 blocks, only PERMUTED by priority. So any KLD
# difference between weights is pure placement, matched-budget. Plus a random floor.
#
# Decisive: does any weight>0 beat weight=0 (positional baseline)? If yes, the model's
# content signal carries KLD-relevant information beyond position, and the winning
# weight is the price to charge it.

import argparse
import struct
import os
import math
import random

PFI1_MAGIC = 0x31494650
RCM1_MAGIC = 0x52434d31

T8 = 8
T4 = 4
T3 = 3
BPW = {8: 1.015625, 4: 0.515625, 3: 0.4375, 16: 16.0}


def read_pfi1(path):
	with open(path, "rb") as f:
		hdr = f.read(20)
		magic, n_windows, n_ctx, block_sz, n_blocks = struct.unpack("<5i", hdr)
		if magic != PFI1_MAGIC:
			raise SystemExit("bad PFI1 magic in %s" % path)
		raw = f.read(n_windows * n_blocks * 4)
	imp = struct.unpack("<%df" % (n_windows * n_blocks), raw)
	grid = [list(imp[w * n_blocks:(w + 1) * n_blocks]) for w in range(n_windows)]
	return n_windows, n_ctx, block_sz, n_blocks, grid


def write_rcm1(path, n_windows, n_ctx, block_sz, block_tiers):
	# block_tiers[w] is a list[n_blocks] of tier codes; expand to per-position bytes.
	with open(path, "wb") as f:
		f.write(struct.pack("<3i", RCM1_MAGIC, n_windows, n_ctx))
		for w in range(n_windows):
			row = bytearray(n_ctx)
			for b, t in enumerate(block_tiers[w]):
				p0 = b * block_sz
				p1 = min(p0 + block_sz, n_ctx)
				for p in range(p0, p1):
					row[p] = t
			f.write(bytes(row))


def zscore(xs):
	m = sum(xs) / len(xs)
	v = sum((x - m) ** 2 for x in xs) / len(xs)
	s = math.sqrt(v) if v > 0 else 1.0
	return [(x - m) / s for x in xs]


def assign_tiers(order, n_blocks, n8, n4):
	# order = block indices ranked best-first; best n8 -> t8, next n4 -> t4, rest -> t3.
	tiers = [T3] * n_blocks
	for rank, b in enumerate(order):
		if rank < n8:
			tiers[b] = T8
		elif rank < n8 + n4:
			tiers[b] = T4
	return tiers


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--imp", required=True, help="PFI1 importance dump")
	ap.add_argument("-o", "--outdir", required=True)
	ap.add_argument("--n8", type=int, default=15, help="blocks at t8")
	ap.add_argument("--n4", type=int, default=20, help="blocks at t4 (rest -> t3)")
	ap.add_argument("--weights", default="0,0.5,1,2,4,8", help="content price sweep")
	ap.add_argument("--seed", type=int, default=1905)
	args = ap.parse_args()

	n_windows, n_ctx, block_sz, n_blocks, grid = read_pfi1(args.imp)
	os.makedirs(args.outdir, exist_ok=True)

	n8, n4 = args.n8, args.n4
	n3 = n_blocks - n8 - n4
	if n3 < 0:
		raise SystemExit("n8+n4 %d exceeds n_blocks %d" % (n8 + n4, n_blocks))
	bpw = (n8 * BPW[8] + n4 * BPW[4] + n3 * BPW[3]) / n_blocks

	# Per-query importance: divide out the causal-summation count (block b sees n_blocks-b
	# query blocks), leaving attention-received-per-query. base = positional profile;
	# resid = per-window content above that profile.
	norm = [[grid[w][b] / (n_blocks - b) for b in range(n_blocks)] for w in range(n_windows)]
	base = [sum(norm[w][b] for w in range(n_windows)) / n_windows for b in range(n_blocks)]
	zpos = zscore(base)
	flat_resid = [norm[w][b] - base[b] for w in range(n_windows) for b in range(n_blocks)]
	m = sum(flat_resid) / len(flat_resid)
	v = sum((x - m) ** 2 for x in flat_resid) / len(flat_resid)
	rs = math.sqrt(v) if v > 0 else 1.0
	zres = [[(norm[w][b] - base[b] - m) / rs for b in range(n_blocks)] for w in range(n_windows)]

	weights = [float(x) for x in args.weights.split(",")]
	print("n_windows=%d n_blocks=%d  hist=%d t8 / %d t4 / %d t3  eff_bpw=%.3f" % (
		n_windows, n_blocks, n8, n4, n3, bpw))
	print("content price sweep weights = %s  (+ random floor)" % weights)

	for wt in weights:
		rows = []
		for w in range(n_windows):
			prio = [zpos[b] + wt * zres[w][b] for b in range(n_blocks)]
			order = sorted(range(n_blocks), key=lambda b: (-prio[b], b))
			rows.append(assign_tiers(order, n_blocks, n8, n4))
		path = os.path.join(args.outdir, "cmask_w%g.bin" % wt)
		write_rcm1(path, n_windows, n_ctx, block_sz, rows)
		print("wrote", path)

	# random floor: same histogram, random permutation
	rows = []
	for w in range(n_windows):
		rng = random.Random(args.seed + w)
		order = list(range(n_blocks))
		rng.shuffle(order)
		rows.append(assign_tiers(order, n_blocks, n8, n4))
	path = os.path.join(args.outdir, "cmask_random.bin")
	write_rcm1(path, n_windows, n_ctx, block_sz, rows)
	print("wrote", path)


if __name__ == "__main__":
	main()
