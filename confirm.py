"""Confirmatory study: 10 fresh seeds (10-19), 3 development + 4 held-out tasks,
n in {10, 30, 2000}, 6 models. Results appended as JSON lines; resumable."""
import json, os, sys, time, numpy as np
from sklearn.metrics import roc_auc_score
from data import load_pair
from baselines import run_logreg, run_hog_svm, run_cnn
from cca2 import CCA2, shift_aug
from ferns import RandomFerns

TASKS = {  # name: (file, class a -> label 0, class b -> label 1, role)
 'MNIST 3v8': ('mnist.npz', 3, 8, 'dev'), 'MNIST 4v9': ('mnist.npz', 4, 9, 'dev'),
 'Fashion T-shirt v Shirt': ('fmnist.npz', 0, 6, 'dev'),
 'MNIST 7v9': ('mnist.npz', 7, 9, 'held-out'), 'MNIST 5v8': ('mnist.npz', 5, 8, 'held-out'),
 'Fashion Pullover v Coat': ('fmnist.npz', 2, 4, 'held-out'),
 'Fashion Sneaker v Ankle boot': ('fmnist.npz', 7, 9, 'held-out')}
SEEDS = list(range(10, 20)); SIZES = [10, 30, 2000]
OUT = 'results/results_confirm.jsonl'

def ev(p, y): return dict(acc=float(((p > 0.5) == y).mean()), auc=float(roc_auc_score(y, p)))
def m_cca2(Xtr, ytr, Xte, yte, s): return ev(CCA2(seed=s).fit(Xtr, ytr).predict_proba(Xte), yte)
def m_ferns(Xtr, ytr, Xte, yte, s):
    if len(Xtr) < 1000: Xtr, ytr = shift_aug(Xtr, ytr, np.random.default_rng(s))
    return ev(RandomFerns(500, 12, seed=s).fit(Xtr, ytr).predict_proba(Xte), yte)
def m_cnn(Xtr, ytr, Xte, yte, s): return run_cnn(Xtr, ytr, Xte, yte, s, epochs=5 if len(Xtr) >= 1000 else 30)
def m_cnn_aug(Xtr, ytr, Xte, yte, s):
    full = len(Xtr) >= 1000
    Xa, ya = shift_aug(Xtr, ytr, np.random.default_rng(s), reps=1 if full else 4)
    return run_cnn(Xa, ya, Xte, yte, s, epochs=6 if full else 30)
MODELS = {'CCA v2': m_cca2, 'LogReg': run_logreg, 'HOG+SVM': run_hog_svm,
          'Random ferns': m_ferns, 'CNN': m_cnn, 'CNN+aug': m_cnn_aug}

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        r = json.loads(l); done.add((r['task'], r['n'], r['seed'], r['model']))
for n in SIZES:
    for task, (f, a, b, role) in TASKS.items():
        for s in (SEEDS if n < 2000 else SEEDS[:5]):          # full data: 5 seeds (compute budget)
            data = None
            for mname, fn in MODELS.items():
                if n == 2000 and mname == 'CNN+aug': continue    # superseded by CNN-strong at full data
                if (task, n, s, mname) in done: continue
                if data is None: data = load_pair(f, a, b, n, 500, s)
                t0 = time.time(); r = fn(*data, s)
                rec = dict(task=task, role=role, n=n, seed=s, model=mname, acc=r['acc'], auc=float(r['auc']),
                           sec=round(time.time() - t0, 2))
                with open(OUT, 'a') as fh: fh.write(json.dumps(rec) + '\n')
print('ALL DONE', flush=True)
