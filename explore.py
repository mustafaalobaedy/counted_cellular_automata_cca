"""Exploratory study (development tasks, seeds 0-2), ablation and development checks.
  main     : LogReg, HOG+SVM, CNN, CNN+aug (n <= 30 only) and CCA v2 at n in {10, 30, 2000}
  ablation : ECA (stage 0 only), ECA + fixed CA rule, CCA v1, CCA v2 without / with augmentation, n in {30, 2000}
  dev      : CCA v1 with and without residual updates (n = 10), CCA v2 with concatenated fusion at n = 10
             and with family-averaged fusion at n = 2000
Results appended as JSON lines to results/results_explore.jsonl; resumable."""
import json, os, time, numpy as np
from sklearn.metrics import roc_auc_score
from data import load_pair, TASKS
from run_experiments import MODELS as MAIN
from cca import CCA, Stage0, pool, shift, DIRS4
from cca2 import CCA2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = [0, 1, 2]
OUT = 'results/results_explore.jsonl'

class FixedRuleCCA:
    """ECA followed by a hand-designed automaton rule instead of learned ones:
    E <- tanh(E + 0.6 * intensity-weighted mean of the 4 neighbours' E), 4 iterations.
    The vote uses the mean of the final map at each scale (the rule replaces the learned stages)."""
    def __init__(s, scales=(1, 2), iters=4, gain=0.6, C=0.1, seed=0):
        s.scales, s.iters, s.gain, s.C, s.seed = scales, iters, gain, C, seed
    def _rule(s, E, X):
        for _ in range(s.iters):
            num = sum(shift(X, dy, dx) * shift(E, dy, dx) for dy, dx in DIRS4)
            den = sum(shift(X, dy, dx) for dy, dx in DIRS4) + 1e-9
            E = np.tanh(E + s.gain * num / den)
        return E
    def _f(s, E, X): return s._rule(E, X).mean((1, 2))[:, None]
    def fit(s, X, y):
        s.s0 = []; F = []
        for f in s.scales:
            Xf = pool(X, f); s.s0.append(Stage0().fit(Xf, y))
            F.append(s._f(_oof0(Xf, y, s.seed), Xf))                     # fusion trained on out-of-fold evidence
        s.cal = make_pipeline(StandardScaler(), LogisticRegression(C=s.C, max_iter=1000)).fit(np.concatenate(F, 1), y)
        return s
    def predict_proba(s, X):
        F = [s._f(m(pool(X, f)), pool(X, f)) for m, f in zip(s.s0, s.scales)]
        return s.cal.predict_proba(np.concatenate(F, 1))[:, 1]

def _oof0(X, y, seed, folds=5):
    """out-of-fold stage-0 evidence maps, same folds as CCAScale"""
    fold = np.random.default_rng(seed).permutation(len(X)) % folds; E = np.zeros(X.shape)
    for k in range(folds):
        tr, te = fold != k, fold == k; E[te] = Stage0().fit(X[tr], y[tr])(X[te])
    return E

def ev(m, Xtr, ytr, Xte, yte):
    p = m.fit(Xtr, ytr).predict_proba(Xte)
    return dict(acc=float(((p > 0.5) == yte).mean()), auc=float(roc_auc_score(yte, p)))

ABL = {'ECA':                lambda s: CCA(stages=0, seed=s),
       'ECA + fixed CA rule': lambda s: FixedRuleCCA(seed=s),
       'CCA v1':             lambda s: CCA(stages=2, seed=s),
       'CCA v2 (no aug)':    lambda s: CCA2(seed=s, aug=0),
       'CCA v2':             lambda s: CCA2(seed=s)}
DEV = {(10, 'CCA v1'):                      lambda s: CCA(stages=2, seed=s),
       (10, 'CCA v1 (no residual)'):       lambda s: CCA(stages=2, seed=s, residual=False),
       (10, 'CCA v2 (concatenated fusion)'): lambda s: CCA2(seed=s, fuse='cat', C=1.0),
       (2000, 'CCA v2 (averaged fusion)'):   lambda s: CCA2(seed=s, fuse='avg', C=0.1)}

if __name__ == '__main__':
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l); done.add((r['study'], r['task'], r['n'], r['seed'], r['model']))
    jobs = []
    for n in (10, 30, 2000):
        for task in TASKS:
            for s in SEEDS:
                for m in MAIN:
                    if not (m == 'cnn_aug' and n == 2000): jobs.append(('main', task, n, s, m, lambda d, s, m=m: MAIN[m](*d, s)))
                for m, mk in ABL.items():
                    if n in (30, 2000) and not (m == 'CCA v2 (no aug)' and n == 2000):
                        jobs.append(('ablation', task, n, s, m, lambda d, s, mk=mk: ev(mk(s), *d)))
                for (nn, m), mk in DEV.items():
                    if nn == n: jobs.append(('dev', task, n, s, m, lambda d, s, mk=mk: ev(mk(s), *d)))
    for study, task, n, s, m, fn in jobs:
        if (study, task, n, s, m) in done: continue
        f, a, b = TASKS[task]; d = load_pair(f, a, b, n, 500, s)
        t0 = time.time(); r = fn(d, s)
        rec = dict(study=study, task=task, n=n, seed=s, model=m, acc=float(r['acc']), auc=float(r['auc']),
                   sec=round(time.time() - t0, 2))
        with open(OUT, 'a') as fh: fh.write(json.dumps(rec) + '\n')
        print(study, task, n, s, m, round(rec['acc'], 3), flush=True)
    print('ALL DONE', flush=True)
