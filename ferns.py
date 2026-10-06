"""Random ferns (Ozuysal et al., 2010) for whole-image binary classification.
Each fern = S random pixel-pair comparisons -> 2^S leaves; class-conditional leaf
probabilities by counting (Laplace alpha); semi-naive Bayes sum of log-odds over ferns."""
import numpy as np

class RandomFerns:
    def __init__(s, n_ferns=50, depth=10, alpha=1.0, seed=0):
        s.F, s.S, s.alpha, s.seed = n_ferns, depth, alpha, seed
    def _leaves(s, X):
        Xf = X.reshape(len(X), -1)
        bits = Xf[:, s.a] > Xf[:, s.b]                       # (n, F, S)
        return (bits * (1 << np.arange(s.S))).sum(-1)        # (n, F)
    def fit(s, X, y):
        rng = np.random.default_rng(s.seed); d = X[0].size
        s.a = rng.integers(0, d, (s.F, s.S)); s.b = rng.integers(0, d, (s.F, s.S))
        L = s._leaves(X); s.T = np.zeros((s.F, 1 << s.S))
        cnt = np.zeros((2, s.F, 1 << s.S))
        for c in (0, 1):
            for f in range(s.F): cnt[c, f] = np.bincount(L[y == c, f], minlength=1 << s.S)
        p = (cnt + s.alpha) / (cnt.sum(-1, keepdims=True) + s.alpha * (1 << s.S))
        s.T = np.log(p[1]) - np.log(p[0]); s.prior = np.log((y == 1).mean() / (y == 0).mean())
        return s
    def decision(s, X):
        L = s._leaves(X); return s.T[np.arange(s.F)[None], L].sum(1) + s.prior
    def predict_proba(s, X):
        return 1 / (1 + np.exp(-np.clip(s.decision(X), -50, 50)))
