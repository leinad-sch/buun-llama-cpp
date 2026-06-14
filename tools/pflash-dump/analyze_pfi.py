#!/usr/bin/env python3
# analyze_pfi.py — sanity-check a PFI1 importance dump (block or per-token).
# Confirms: no NaN/Inf, the causal front-load is present (raw imp ~ scales with the
# number of causal queries T-pos), and the per-query-normalized POSITION profile +
# per-window CONTENT residual are well-conditioned (not degenerate / not all-zero).
import struct, sys, math

PFI1_MAGIC = 0x31494650

def read_pfi1(path):
	with open(path, "rb") as f:
		magic, nw, nctx, bsz, nu = struct.unpack("<5i", f.read(20))
		if magic != PFI1_MAGIC:
			raise SystemExit("bad magic")
		raw = f.read(nw * nu * 4)
	imp = struct.unpack("<%df" % (nw * nu), raw)
	grid = [list(imp[w*nu:(w+1)*nu]) for w in range(nw)]
	return nw, nctx, bsz, nu, grid

def stats(xs):
	n = len(xs); m = sum(xs)/n
	v = sum((x-m)**2 for x in xs)/n
	return m, math.sqrt(v), min(xs), max(xs)

def main():
	path = sys.argv[1]
	nw, nctx, bsz, nu, grid = read_pfi1(path)
	print("file=%s  n_windows=%d n_ctx=%d block_sz=%d n_units=%d  (%s)" % (
		path, nw, nctx, bsz, nu, "PER-TOKEN" if bsz == 1 else "per-%d-block" % bsz))

	flat = [grid[w][b] for w in range(nw) for b in range(nu)]
	nan = sum(1 for x in flat if x != x)
	inf = sum(1 for x in flat if x == float("inf") or x == float("-inf"))
	m, s, lo, hi = stats(flat)
	print("RAW   mean=%.3f std=%.3f min=%.3f max=%.3f  NaN=%d Inf=%d" % (m, s, lo, hi, nan, inf))

	# causal front-load: raw imp[pos] should fall roughly as (T_units - pos).
	# report raw mean over windows at a few positions (front / mid / wall).
	def col_mean(b): return sum(grid[w][b] for w in range(nw))/nw
	probes = [0, nu//4, nu//2, 3*nu//4, nu-1]
	print("RAW per-pos mean @ idx", probes, "=",
		["%.1f" % col_mean(b) for b in probes])

	# per-query normalized (divide out causal count T_units - pos)
	norm = [[grid[w][b]/(nu-b) for b in range(nu)] for w in range(nw)]
	base = [sum(norm[w][b] for w in range(nw))/nw for b in range(nu)]
	bm, bs, blo, bhi = stats(base)
	print("NORM position profile: mean=%.4f std=%.4f min=%.4f max=%.4f  (front=%.4f wall=%.4f ratio=%.2fx)" % (
		bm, bs, blo, bhi, base[0], base[-1], (base[-1]/base[0] if base[0] else 0)))

	resid = [norm[w][b]-base[b] for w in range(nw) for b in range(nu)]
	rm, rs, rlo, rhi = stats(resid)
	# content-to-position ratio: how big is per-window content swing vs positional swing?
	print("NORM content residual: std=%.5f  (vs position std=%.5f -> content/position=%.2f)" % (
		rs, bs, (rs/bs if bs else 0)))
	# fraction of variance that is content (within-window) vs position (across-pos)
	tot = sum((norm[w][b]-bm)**2 for w in range(nw) for b in range(nu))
	posvar = nw*sum((base[b]-bm)**2 for b in range(nu))
	print("variance split: position=%.1f%%  content=%.1f%%" % (
		100*posvar/tot if tot else 0, 100*(1-posvar/tot) if tot else 0))

if __name__ == "__main__":
	main()
