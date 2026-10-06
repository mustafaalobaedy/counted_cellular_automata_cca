"""Run CCA v2 and baselines.  Usage:
   python run_experiments.py <n_per_class> <task name> <model>
   model in {cca2, logreg, hogsvm, cnn, cnn_aug}"""
import sys, time, numpy as np
from sklearn.metrics import roc_auc_score
from data import load_pair, TASKS
from baselines import run_logreg, run_hog_svm, run_cnn
from cca2 import CCA2, shift_aug

SEEDS = [0, 1, 2]

def run_cca2(Xtr, ytr, Xte, yte, seed):
    p = CCA2(seed=seed).fit(Xtr, ytr).predict_proba(Xte)       # 'auto' configuration
    return dict(acc=((p > 0.5) == yte).mean(), auc=roc_auc_score(yte, p))

def run_cnn_aug(Xtr, ytr, Xte, yte, seed):
    Xa, ya = shift_aug(Xtr, ytr, np.random.default_rng(seed))   # same augmentation as CCA
    return run_cnn(Xa, ya, Xte, yte, seed, epochs=30)

MODELS = {'cca2': run_cca2, 'logreg': run_logreg, 'hogsvm': run_hog_svm,
          'cnn': lambda *a: run_cnn(*a, epochs=5 if len(a[0]) >= 1000 else 30),
          'cnn_aug': run_cnn_aug}

if __name__ == '__main__':
    n, task, model = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    file, a, b = TASKS[task]; rs = []; t0 = time.time()
    for s in SEEDS:
        rs.append(MODELS[model](*load_pair(file, a, b, n, 500, s), s))
    acc = [r['acc'] for r in rs]; auc = [r['auc'] for r in rs]
    print(f"n={n} {task} {model}: acc={np.mean(acc):.4f}±{np.std(acc):.3f} "
          f"auc={np.mean(auc):.4f}±{np.std(auc):.3f} ({(time.time()-t0)/len(SEEDS):.1f}s/seed)")
