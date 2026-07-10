#!/usr/bin/env python3
# E9 offline: does a data-aware rotation beat the fixed FWHT for 1-bit sign quant on real KV?
# Invert the post-FWHT harvest to raw unit vectors, then sign-quantize under different rotations.
import numpy as np
from scipy.linalg import hadamard

S1 = np.array([-1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1],dtype=np.float64)
S2 = np.array([1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1],dtype=np.float64)
H = hadamard(128).astype(np.float64)
INV_SQRT = 1.0/np.sqrt(128.0)

def fwht_fwd(x):   # forward turbo rotation: signs2 * FWHT_norm(signs1 * x)  (x: [N,128])
    return (S2 * (INV_SQRT * (S1*x) @ H.T))
def fwht_inv(x):   # inverse: signs1 * FWHT_norm(signs2 * x)
    return (S1 * (INV_SQRT * (S2*x) @ H.T))

post = np.fromfile('/tmp/turbo_postrot.bin', dtype=np.float32).reshape(-1,128).astype(np.float64)
post = post[:40000]
raw = fwht_inv(post)                      # unit raw vectors (||raw||~1)
raw /= np.linalg.norm(raw,axis=1,keepdims=True)+1e-12
N = raw.shape[0]
print(f"N={N}  mean||raw||={np.linalg.norm(raw,axis=1).mean():.4f}")

# verify round-trip: sign(fwht_fwd(raw)) reconstruct should match the deployed codec
def sign_quant(xr):                        # xr rotated [N,128]; sign + norm-preserving sigma
    q = np.sign(xr); q[q==0]=1
    return q*INV_SQRT                       # recon in rotated domain (||recon||=1 per row)

def eval_rot(name, Rfwd, Rinv):
    xr = Rfwd(raw)
    recon = Rinv(sign_quant(xr))           # back to original domain
    rel_recon = (np.linalg.norm(recon-raw,axis=1)**2).mean()  # V proxy (recon MSE, ||raw||=1)
    # K proxy: random Gaussian queries, dot-product error
    rng=np.random.default_rng(0); Q=rng.standard_normal((256,128))
    Q/=np.linalg.norm(Q,axis=1,keepdims=True)+1e-12
    true=raw@Q.T; est=recon@Q.T
    rel_dot=((est-true)**2).mean()/ (true**2).mean()
    print(f"{name:16s} recon_MSE={rel_recon:.4f}  rel_dot_err={rel_dot:.4f}")
    return rel_recon, rel_dot

# Rotations (all orthonormal, applied to unit raw):
eval_rot("FWHT(deployed)", fwht_fwd, fwht_inv)
# random orthogonal (different basis, data-oblivious)
rng=np.random.default_rng(1); Rr,_=np.linalg.qr(rng.standard_normal((128,128)))
eval_rot("random-orth", lambda x:x@Rr.T, lambda x:x@Rr)
# PCA: eigenvectors of raw covariance (data-aware, concentrates energy)
C=(raw.T@raw)/N; w,Vp=np.linalg.eigh(C)
eval_rot("PCA(raw cov)", lambda x:x@Vp.T, lambda x:x@Vp)
# FWHT then PCA-of-postFWHT (align to residual structure FWHT leaves)
Cp=(post.T@post)/post.shape[0]; wp,Vpp=np.linalg.eigh(Cp/ (np.trace(Cp)/128))
eval_rot("FWHT+PCA", lambda x:(S2*(INV_SQRT*(S1*x)@H.T))@Vpp.T, lambda x:(S1*(INV_SQRT*(S2*(x@Vpp))@H.T)))
print("(lower is better; FWHT is the bar to beat)")
