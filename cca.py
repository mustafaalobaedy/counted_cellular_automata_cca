"""
Counted Cellular Automaton (CCA) — binary image classifier.
No convolution, no gradients. Every component is learned by counting.

Stage 0  : pixel code (8 neighbour-vs-centre bits + ink bit, 512 codes)
           -> position-specific evidence LUT (blurred counts, naive-Bayes log-odds)
Stage k>0: the CA update rule is itself a LUT learned by counting:
           state(u) = (ternary(e_{k-1}) at u and its 4 neighbours, ink bit at u)  -> 486 states
           e_k(u)   = R_k[region(u), state(u)]      (log-odds, region = 7x7 grid)
           Training inputs e_{k-1} are produced out-of-fold (K-fold cross-fitting)
           so each rule learns from honest, not memorised, evidence.
Vote     : per stage & scale, mean evidence -> fused by logistic calibration
           trained on out-of-fold features.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegression

DIRS8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
DIRS4 = [(-1,0),(0,-1),(0,1),(1,0)]

def shift(X, dy, dx):
    P = np.pad(X, ((0,0),(1,1),(1,1)), mode='edge'); h, w = X.shape[1:]
    return P[:, 1+dy:1+dy+h, 1+dx:1+dx+w]

def pix_codes(X, thr=0.1):
    c = (X > 0.3).astype(np.int64) << 8
    for b,(dy,dx) in enumerate(DIRS8):
        c |= ((shift(X,dy,dx) > X + thr).astype(np.int64) << b)
    return c

def shift2(X, dy, dx):
    P = np.pad(X, ((0,0),(2,2),(2,2)), mode='edge'); h, w = X.shape[1:]
    return P[:, 2+dy:2+dy+h, 2+dx:2+dx+w]

def ring2_codes(X, thr=0.1):
    c = (X > 0.3).astype(np.int64) << 8
    for b,(dy,dx) in enumerate(DIRS8):
        c |= ((shift2(X,2*dy,2*dx) > X + thr).astype(np.int64) << b)
    return c                                                   # 512 codes

def grad_codes(X):
    gy = shift(X,1,0) - shift(X,-1,0); gx = shift(X,0,1) - shift(X,0,-1)
    mag = np.hypot(gx, gy); ang = np.arctan2(gy, gx)
    o = ((ang + np.pi) / (2*np.pi) * 8).astype(np.int64) % 8
    m = (mag > 0.1).astype(np.int64) + (mag > 0.5)
    return ((o*3 + m) << 1) | (X > 0.3)                        # 48 codes

CODE_FAMILIES = {'ring1': (pix_codes, 512), 'ring2': (ring2_codes, 512), 'grad': (grad_codes, 48)}

def pool(X, f):
    n,h,w = X.shape; return X.reshape(n,h//f,f,w//f,f).mean((2,4))

def lut_fit(pos, code, y, npos, ncode, alpha, blur_shape=None, blur=0.0):
    """log-odds table [npos, ncode] from counts; pos/code: (n, m) int arrays"""
    cnt = np.zeros((2, npos, ncode))
    for c in (0,1):
        m = y == c
        np.add.at(cnt[c], (pos[m].ravel(), code[m].ravel()), 1)
        if blur > 0:
            g = cnt[c].reshape(*blur_shape, ncode)
            cnt[c] = gaussian_filter(g, (blur, blur, 0)).reshape(npos, ncode)
    p = (cnt + alpha) / (cnt.sum(-1, keepdims=True) + ncode*alpha)
    return np.log(p[1]) - np.log(p[0])

class Stage0:
    """sum of per-family position-specific log-odds LUTs (naive Bayes over code families)"""
    def __init__(s, blur=1.0, alpha=0.5, families=('ring1',)):
        s.blur, s.alpha, s.fam = blur, alpha, families
    def fit(s, X, y):
        n,h,w = X.shape; s.h, s.w = h, w
        pos = np.broadcast_to(np.arange(h*w).reshape(h,w), X.shape)
        s.L = {f: lut_fit(pos, CODE_FAMILIES[f][0](X), y, h*w, CODE_FAMILIES[f][1], s.alpha, (h,w), s.blur)
               for f in s.fam}
        return s
    def __call__(s, X):
        P = np.arange(s.h*s.w).reshape(s.h,s.w)[None]; E = 0
        for f in s.fam:
            fn, K = CODE_FAMILIES[f]; E = E + s.L[f].ravel()[P*K + fn(X)]
        return E

class RuleStage:
    """learned CA rule: e_k(u) = R[region(u), state(u)]"""
    def __init__(s, grid=7, alpha=1.0): s.grid, s.alpha = grid, alpha
    def _state(s, E, X):
        t = s.t
        q = lambda A: (A > t).astype(np.int64) - (A < -t) + 1          # ternary 0/1/2
        st = q(E)
        for dy,dx in DIRS4: st = st*3 + q(shift(E,dy,dx))
        return st*2 + (X > 0.3)                                        # 3^5*2 = 486 states
    def _region(s, shape):
        n,h,w = shape
        r = (np.arange(h)[:,None]*s.grid//h)*s.grid + np.arange(w)[None]*s.grid//w
        return np.broadcast_to(r, shape)
    def fit(s, E, X, y):
        s.t = np.percentile(np.abs(E), 60) + 1e-9
        s.R = lut_fit(s._region(E.shape), s._state(E, X), y, s.grid**2, 486, s.alpha); return s
    def __call__(s, E, X):
        return s.R.ravel()[s._region(E.shape)*486 + s._state(E, X)]

class CCAScale:
    def __init__(s, stages=2, folds=5, seed=0, residual=True, **kw):  # residual=True, C=0.1 = best config
        s.stages, s.folds, s.seed, s.res, s.kw = stages, folds, seed, residual, kw
    def fit(s, X, y, groups=None):
        """cross-fitted training: out-of-fold evidence at every stage.
        groups: original-image id per row, so augmented copies share a fold"""
        n = len(X); rng = np.random.default_rng(s.seed)
        if groups is None: groups = np.arange(n)
        fold = (rng.permutation(groups.max()+1) % s.folds)[groups]
        oof = [np.zeros(X.shape)]                               # stage-0 OOF evidence
        for f in range(s.folds):
            tr, te = fold != f, fold == f
            oof[0][te] = Stage0(**s.kw).fit(X[tr], y[tr])(X[te])
        s.s0 = Stage0(**s.kw).fit(X, y)
        s.rules = []
        for k in range(s.stages):
            prev = oof[-1]; nxt = np.zeros(X.shape)
            for f in range(s.folds):
                tr, te = fold != f, fold == f
                nxt[te] = RuleStage().fit(prev[tr], X[tr], y[tr])(prev[te], X[te]) + (prev[te] if s.res else 0)
            s.rules.append(RuleStage().fit(prev, X, y))
            oof.append(nxt)
        s.oof_feats = np.stack([e.mean((1,2)) for e in oof], 1)
        return s
    def feats(s, X):
        E = s.s0(X); out = [E.mean((1,2))]
        for r in s.rules:
            E = r(E, X) + (E if s.res else 0); out.append(E.mean((1,2)))
        return np.stack(out, 1)

def shift_aug(X, y):
    Xs=[X]; ys=[y]
    for dy,dx in DIRS4:
        Xs.append(np.roll(np.roll(X,dy,1),dx,2)); ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)

class CCA:
    def __init__(s, scales=(1,2), stages=2, C=0.1, aug=False, **kw):
        s.scales, s.stages, s.C, s.aug, s.kw = scales, stages, C, aug, kw
    def fit(s, X, y):
        g = np.arange(len(X))
        if s.aug: X, y = shift_aug(X, y); g = np.tile(g, len(X)//len(g))
        s.models = [CCAScale(stages=s.stages, **s.kw).fit(pool(X,f), y, g) for f in s.scales]
        F = np.concatenate([m.oof_feats for m in s.models], 1)
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        s.cal = make_pipeline(StandardScaler(), LogisticRegression(C=s.C, max_iter=1000)).fit(F, y); return s
    def feats(s, X): return np.concatenate([m.feats(pool(X,f)) for m,f in zip(s.models, s.scales)], 1)
    def predict_proba(s, X): return s.cal.predict_proba(s.feats(X))[:,1]
