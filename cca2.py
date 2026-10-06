"""
CCA v2 — Counted Cellular Automaton with multiple code families.
Everything is still learned by counting (no gradients, no convolution).

Code families (stage-0 pixel codes):
  'cmp1' : 8 neighbour-vs-centre bits at radius 1 + ink bit          (512 codes)
  'cmp2' : same comparison at radius 2                                (512 codes)
  'ori'  : gradient orientation (8 bins) x magnitude (2) x ink (2)    (32 codes)
Each family has its own stage-0 LUT + learned CA rule chain (cross-fitted).
Fusion: per (scale, stage), scores are averaged across families (fixed weights),
then a small regularised logistic layer fuses scales x stages.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from cca import lut_fit, RuleStage, pool

DIRS8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def shiftr(X, dy, dx, r=1):
    P = np.pad(X, ((0,0),(r,r),(r,r)), mode='edge'); h, w = X.shape[1:]
    return P[:, r+dy:r+dy+h, r+dx:r+dx+w]

def code_cmp(X, r, thr=0.1):
    c = (X > 0.3).astype(np.int64) << 8
    for b,(dy,dx) in enumerate(DIRS8):
        c |= ((shiftr(X, dy*r, dx*r, r) > X + thr).astype(np.int64) << b)
    return c

def code_ori(X):
    gy = shiftr(X,1,0) - shiftr(X,-1,0); gx = shiftr(X,0,1) - shiftr(X,0,-1)
    ang = ((np.arctan2(gy, gx) + np.pi) / (2*np.pi) * 8).astype(np.int64) % 8
    mag = (np.hypot(gx, gy) > 0.2).astype(np.int64)
    return (ang*2 + mag)*2 + (X > 0.3)

FAMILIES = {'cmp1': (lambda X: code_cmp(X,1), 512),
            'cmp2': (lambda X: code_cmp(X,2), 512),
            'ori':  (code_ori, 32)}

class Stage0F:
    def __init__(s, fam, blur=1.0, alpha=0.5):
        s.fn, s.nc = FAMILIES[fam]; s.blur, s.alpha = blur, alpha
    def fit(s, X, y):
        n,h,w = X.shape; s.h, s.w = h, w
        pos = np.broadcast_to(np.arange(h*w).reshape(h,w), X.shape)
        s.L = lut_fit(pos, s.fn(X), y, h*w, s.nc, s.alpha, (h,w), s.blur); return s
    def __call__(s, X):
        return s.L.ravel()[np.arange(s.h*s.w).reshape(s.h,s.w)[None]*s.nc + s.fn(X)]

def shift_aug(X, y, rng, reps=4, s=2):
    Xs=[X]; ys=[y]
    for _ in range(reps):
        d = rng.integers(-s, s+1, (len(X),2))
        Xs.append(np.stack([np.roll(np.roll(im,a,0),b,1) for im,(a,b) in zip(X,d)])); ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)

class ChainF:
    """one family at one scale: stage-0 LUT + residual learned CA rules, cross-fitted"""
    def __init__(s, fam, stages=2, folds=5, seed=0, aug=0):
        s.fam, s.stages, s.folds, s.seed, s.aug = fam, stages, folds, seed, aug
    def fit(s, X, y):
        n = len(X); rng = np.random.default_rng(s.seed)
        fold = rng.permutation(n) % s.folds
        A = (lambda X_, y_: shift_aug(X_, y_, rng, s.aug)) if s.aug else (lambda X_, y_: (X_, y_))
        prev = np.zeros(X.shape)
        for f in range(s.folds):
            tr, te = fold != f, fold == f
            prev[te] = Stage0F(s.fam).fit(*A(X[tr], y[tr]))(X[te])
        s.s0 = Stage0F(s.fam).fit(*A(X, y))
        feats = [prev.mean((1,2))]; s.rules = []
        for _ in range(s.stages):
            nxt = np.zeros(X.shape)
            for f in range(s.folds):
                tr, te = fold != f, fold == f
                nxt[te] = RuleStage().fit(prev[tr], X[tr], y[tr])(prev[te], X[te]) + prev[te]
            s.rules.append(RuleStage().fit(prev, X, y))
            prev = nxt; feats.append(prev.mean((1,2)))
        s.oof = np.stack(feats, 1); return s
    def feats(s, X):
        E = s.s0(X); out = [E.mean((1,2))]
        for r in s.rules:
            E = r(E, X) + E; out.append(E.mean((1,2)))
        return np.stack(out, 1)

class CCA2:
    def __init__(s, families=('cmp1','cmp2','ori'), scales=(1,2), stages=2, C='auto',
                 fuse='auto', aug='auto', seed=0):
        # 'auto' = best settings found in experiments:
        #   n/class < 30  : family-averaged fusion, C=0.1, shift aug x4
        #   30 <= n < 500 : concatenated fusion,   C=1.0, shift aug x4
        #   n >= 500      : concatenated fusion,   C=1.0, no aug
        s.fams, s.scales, s.stages, s.C, s.fuse, s.aug, s.seed = families, scales, stages, C, fuse, aug, seed
    def _combine(s, blocks):        # blocks[scale][family] -> (n, stages+1)
        if s.fuse == 'avg':
            return np.concatenate([np.mean(b, 0) for b in blocks], 1)
        return np.concatenate([np.concatenate(b, 1) for b in blocks], 1)
    def fit(s, X, y):
        n = np.bincount(y).min()
        if s.fuse == 'auto': s.fuse = 'avg' if n < 30 else 'cat'
        if s.C == 'auto':    s.C = 0.1 if n < 30 else 1.0
        if s.aug == 'auto':  s.aug = 4 if n < 500 else 0
        s.chains = [[ChainF(fam, s.stages, seed=s.seed, aug=s.aug).fit(pool(X,f), y) for fam in s.fams]
                    for f in s.scales]
        F = s._combine([[c.oof for c in cs] for cs in s.chains])
        s.cal = make_pipeline(StandardScaler(), LogisticRegression(C=s.C, max_iter=2000)).fit(F, y)
        return s
    def predict_proba(s, X):
        F = s._combine([[c.feats(pool(X,f)) for c in cs] for cs,f in zip(s.chains, s.scales)])
        return s.cal.predict_proba(F)[:,1]
