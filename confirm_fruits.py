"""Real-world study on Fruits-360 (fruits and vegetables), run with the CCA design frozen as in the
confirmatory study: same six models, same settings ('auto' CCA), seeds 10-19, no tuning on these data.
Tasks were fixed before any run: visually similar pairs (shape / size), mixing fruits and vegetables.
Sizes: 10 and 30 training images per class (10 seeds) and all available (438-1412/class, balanced to the smaller class; 5 seeds).
Test set: the official Fruits-360 test images of both classes (balanced, fixed across seeds).
Results appended as JSON lines to results/results_fruits.jsonl; resumable."""
import json, os, time, warnings, numpy as np
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from data_fruits import load_pair
from baselines import run_logreg, run_hog_svm, run_cnn, run_cnn_strong
from cca2 import CCA2, shift_aug
from ferns import RandomFerns

TASKS = {  # name: (class a -> label 0, class b -> label 1)
 'Apple Red v Tomato':          ('Apple Red', 'Tomato'),
 'Peach v Nectarine':           ('Peach', 'Nectarine'),
 'Lemon v Limes':               ('Lemon', 'Limes'),
 'Pear v Quince':               ('Pear', 'Quince'),
 'Mandarine v Clementine':      ('Mandarine', 'Clementine'),
 'Onion White v Potato White':  ('Onion White', 'Potato White'),
 'Pepper Green v Pepper Red':   ('Pepper Green', 'Pepper Red')}
SEEDS = list(range(10, 20)); SIZES = [10, 30, 'all']
OUT = 'results/results_fruits.jsonl'

def ev(p, y): return dict(acc=float(((p > 0.5) == y).mean()), auc=float(roc_auc_score(y, p)))
def m_cca2(Xtr, ytr, Xte, yte, s): return ev(CCA2(seed=s).fit(Xtr, ytr).predict_proba(Xte), yte)
def m_ferns(Xtr, ytr, Xte, yte, s):
    if len(Xtr) < 1000: Xtr, ytr = shift_aug(Xtr, ytr, np.random.default_rng(s))
    return ev(RandomFerns(500, 12, seed=s).fit(Xtr, ytr).predict_proba(Xte), yte)
def m_cnn(Xtr, ytr, Xte, yte, s): return run_cnn(Xtr, ytr, Xte, yte, s, epochs=5 if len(Xtr) >= 1000 else 30)
def m_cnn_aug(Xtr, ytr, Xte, yte, s):
    if len(Xtr) >= 1000: return run_cnn_strong(Xtr, ytr, Xte, yte, s)    # as at n=2000 in the confirmatory study
    Xa, ya = shift_aug(Xtr, ytr, np.random.default_rng(s), reps=4)
    return run_cnn(Xa, ya, Xte, yte, s, epochs=30)
MODELS = {'CCA v2': m_cca2, 'LogReg': run_logreg, 'HOG+SVM': run_hog_svm,
          'Random ferns': m_ferns, 'CNN': m_cnn, 'CNN+aug': m_cnn_aug}

if __name__ == '__main__':
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l); done.add((r['task'], r['n'], r['seed'], r['model']))
    for n in SIZES:
        for task, (a, b) in TASKS.items():
            for s in (SEEDS if n != 'all' else SEEDS[:5]):
                data = None
                for mname, fn in MODELS.items():
                    if (task, n, s, mname) in done: continue
                    if data is None: data = load_pair(a, b, n, s)
                    t0 = time.time(); r = fn(*data, s)
                    rec = dict(task=task, role='fruits', n=n, n_per_class=int(len(data[1]) // 2), n_test=int(len(data[3])),
                               seed=s, model=mname, acc=float(r['acc']), auc=float(r['auc']), sec=round(time.time() - t0, 2))
                    with open(OUT, 'a') as fh: fh.write(json.dumps(rec) + '\n')
                print(task, n, s, flush=True)
    print('ALL DONE', flush=True)
