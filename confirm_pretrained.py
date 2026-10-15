"""Revision study (Oct 2026): adds a transfer-learning baseline, ImageNet-pretrained ResNet-18 features +
logistic regression (pretrained.py), to both confirmatory studies of the main CCA paper. Nothing else changes:
the CCA design and the five original baselines are not re-run; their results are read from
results/ by analysis_pretrained.py.

Protocol, fixed before any run and identical to the original runners:
  - MNIST / Fashion-MNIST: the 7 tasks of confirm.py, n in {10, 30, 2000} per class, seeds 10-19
    (10-14 at n = 2000), 500 test images per class, data.load_pair.
  - Fruits-360: the 7 tasks of confirm_fruits.py, n in {10, 30, 'all'}, seeds 10-19 (10-14 at 'all'),
    official test split, data_fruits.load_pair.
Before the study, check_splits() re-runs the deterministic pixel logistic regression on the first seed of every
(task, size) cell and requires its accuracy to equal the recorded one, which shows that the re-downloaded data
give the same training and test images as the original study. Amended on 15 Oct 2026, before any ResNet run:
at the full-data sizes the solver does not reproduce bit for bit on this machine (MNIST/Fashion Pullover v Coat,
n = 2000: 0.863 now vs 0.864 recorded, in both Python environments), so at full data a difference of at most two
test images (0.002) is accepted and logged; at n = 10 and 30 the match must be exact. The full-data images are
prefixes of the same per-class permutations as the small-data ones, so exact small-data matches pin the data.

One JSON line per (task, n, seed) in results/results_pretrained.jsonl; resumable; prints ALL DONE."""
import json, os, time, warnings
warnings.filterwarnings('ignore')
import data, data_fruits
from baselines import run_logreg
from pretrained import run_resnet_probe

CONFIRM = {'MNIST 3v8': ('mnist.npz', 3, 8, 'dev'), 'MNIST 4v9': ('mnist.npz', 4, 9, 'dev'),
 'Fashion T-shirt v Shirt': ('fmnist.npz', 0, 6, 'dev'),
 'MNIST 7v9': ('mnist.npz', 7, 9, 'held-out'), 'MNIST 5v8': ('mnist.npz', 5, 8, 'held-out'),
 'Fashion Pullover v Coat': ('fmnist.npz', 2, 4, 'held-out'),
 'Fashion Sneaker v Ankle boot': ('fmnist.npz', 7, 9, 'held-out')}
FRUITS = {'Apple Red v Tomato': ('Apple Red', 'Tomato'), 'Peach v Nectarine': ('Peach', 'Nectarine'),
 'Lemon v Limes': ('Lemon', 'Limes'), 'Pear v Quince': ('Pear', 'Quince'),
 'Mandarine v Clementine': ('Mandarine', 'Clementine'), 'Onion White v Potato White': ('Onion White', 'Potato White'),
 'Pepper Green v Pepper Red': ('Pepper Green', 'Pepper Red')}
SEEDS = list(range(10, 20))
OUT = 'results/results_pretrained.jsonl'; MODEL = 'ResNet-18 probe'
OLD = {'confirm': 'results/results_confirm.jsonl', 'fruits': 'results/results_fruits.jsonl'}

def cells():
    for n in (10, 30, 2000):
        for task, (f, a, b, role) in CONFIRM.items():
            for s in (SEEDS if n != 2000 else SEEDS[:5]):
                yield 'confirm', task, role, n, s, (lambda f=f, a=a, b=b, n=n, s=s: data.load_pair(f, a, b, n, 500, s))
    for n in (10, 30, 'all'):
        for task, (a, b) in FRUITS.items():
            for s in (SEEDS if n != 'all' else SEEDS[:5]):
                yield 'fruits', task, 'fruits', n, s, (lambda a=a, b=b, n=n, s=s: data_fruits.load_pair(a, b, n, s))

def check_splits():
    rec = {}
    for study, path in OLD.items():
        for l in open(path):
            r = json.loads(l)
            if r['model'] == 'LogReg': rec[(study, r['task'], r['n'], r['seed'])] = r['acc']
    for study, task, role, n, s, load in cells():
        if s != 10: continue
        acc = float(run_logreg(*load(), s)['acc'])
        old = rec[(study, task, n, s)]
        tol = 1e-12 if n in (10, 30) else 0.002 + 1e-12
        assert abs(acc - old) <= tol, f'split mismatch {study} {task} n={n}: {acc} vs recorded {old}'
        print('split ok', study, task, n, '' if acc == old else f'(solver difference {acc - old:+.4f})', flush=True)

if __name__ == '__main__':
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l); done.add((r['study'], r['task'], r['n'], r['seed']))
    else:
        check_splits()
    for study, task, role, n, s, load in cells():
        if (study, task, n, s) in done: continue
        d = load(); t0 = time.time(); r = run_resnet_probe(*d, s)
        out = dict(study=study, task=task, role=role, n=n, n_per_class=int(len(d[1]) // 2), n_test=int(len(d[3])),
                   seed=s, model=MODEL, acc=r['acc'], auc=r['auc'], sec=round(time.time() - t0, 2))
        with open(OUT, 'a') as fh: fh.write(json.dumps(out) + '\n')
        print(study, task, n, s, round(r['acc'], 4), flush=True)
    print('ALL DONE', flush=True)
