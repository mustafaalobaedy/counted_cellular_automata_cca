"""Final statistics for the confirmatory study.
Models: CCA v2, LogReg, HOG+SVM, Random ferns, CNN, CNN-aug, where CNN-aug = CNN+aug (n<=30, 10 seeds)
or CNN-strong (n=2000, 5 seeds). Tests: Wilcoxon signed-rank (Pratt) paired by (task, seed), Holm-corrected;
Friedman over the 21 (task, n) cells with average ranks and Nemenyi CD (Demsar, 2006)."""
import json, numpy as np, collections
from scipy.stats import wilcoxon, friedmanchisquare
MODELS = ['CCA v2', 'LogReg', 'HOG+SVM', 'Random ferns', 'CNN', 'CNN-aug']
acc = collections.defaultdict(dict); auc = collections.defaultdict(dict); role = {}; secs = collections.defaultdict(list)
for l in open('results/results_confirm.jsonl'):
    r = json.loads(l); m = r['model']
    if r['n'] == 2000 and m == 'CNN+aug': continue                 # superseded by CNN-strong
    if m in ('CNN+aug', 'CNN-strong'): m = 'CNN-aug'
    if r['n'] == 2000 and r['seed'] > 14: continue
    key = (r['task'], r['n'], m); role[r['task']] = r['role']
    if r['seed'] not in acc[key]:
        acc[key][r['seed']] = r['acc']; auc[key][r['seed']] = r['auc']; secs[(r['n'], r['model'])].append(r['sec'])
tasks = list(dict.fromkeys(k[0] for k in acc)); sizes = [10, 30, 2000]
need = {10: 10, 30: 10, 2000: 5}
cells = [(t, n) for n in sizes for t in tasks if all(len(acc[(t, n, m)]) == need[n] for m in MODELS)]
mean = lambda d: float(np.mean(list(d.values())))
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for i, j in enumerate(o): run = max(run, min(1.0, (len(p) - i) * p[j])); adj[j] = run
    return adj
def wil(d): return float(wilcoxon(d, zero_method='pratt').pvalue) if np.any(d != 0) else 1.0
res = {'cells': [], 'pooled': [], 'percell': [], 'complete_cells': len(cells)}
for t, n in cells:
    res['cells'].append(dict(task=t, role=role[t], n=n, **{m: dict(acc=mean(acc[(t, n, m)]),
        sd=float(np.std(list(acc[(t, n, m)].values()), ddof=1)), auc=mean(auc[(t, n, m)])) for m in MODELS}))
# pooled: CCA vs each baseline per size, all tasks and held-out only (Holm within each scope)
for scope in ('all', 'held-out'):
    block = []
    for n in sizes:
        cs = [t for t, nn in cells if nn == n and (scope == 'all' or role[t] == 'held-out')]
        for m in MODELS[1:]:
            d = np.array([acc[(t, n, 'CCA v2')][s] - acc[(t, n, m)][s] for t in cs for s in sorted(acc[(t, n, m)])])
            block.append(dict(scope=scope, n=n, baseline=m, pairs=len(d), mean_diff=float(d.mean()),
                              wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum()), p=wil(d)))
    for x, a in zip(block, holm([b['p'] for b in block])): x['p_holm'] = float(a)
    res['pooled'] += block
# per cell: CCA vs each baseline (Holm across all per-cell tests)
pcs = []
for t, n in cells:
    for m in MODELS[1:]:
        d = np.array([acc[(t, n, 'CCA v2')][s] - acc[(t, n, m)][s] for s in sorted(acc[(t, n, m)])])
        pcs.append(dict(task=t, n=n, baseline=m, mean_diff=float(d.mean()), p=wil(d)))
for x, a in zip(pcs, holm([x['p'] for x in pcs])): x['p_holm'] = float(a)
res['percell'] = pcs
# Friedman / average ranks
M = np.array([[mean(acc[(t, n, m)]) for m in MODELS] for t, n in cells])
from scipy.stats import rankdata
R = np.array([rankdata(-row) for row in M]); k, N = len(MODELS), len(cells)
chi2, p = friedmanchisquare(*M.T)
res['friedman'] = dict(chi2=float(chi2), p=float(p), N=N, k=k, avg_rank=dict(zip(MODELS, map(float, R.mean(0)))),
                       nemenyi_CD=float(2.850 * np.sqrt(k * (k + 1) / (6 * N))),
                       **{f'avg_rank_n{n}': dict(zip(MODELS, map(float, R[[i for i, c in enumerate(cells) if c[1] == n]].mean(0)))) for n in sizes},
                       avg_rank_heldout=dict(zip(MODELS, map(float, R[[i for i, c in enumerate(cells) if role[c[0]] == 'held-out']].mean(0)))))
res['best_counts'] = dict(collections.Counter(MODELS[int(np.argmax(row))] for row in M))
res['train_sec_median'] = {f'{n}|{m}': float(np.median(v)) for (n, m), v in secs.items()}
json.dump(res, open('results/final_analysis.json', 'w'), indent=1)
print('complete cells', len(cells), '| best counts', res['best_counts'])
print('Friedman', {k_: v for k_, v in res['friedman'].items() if k_ in ('chi2', 'p', 'N', 'nemenyi_CD')})
print('avg ranks', {m: round(v, 2) for m, v in res['friedman']['avg_rank'].items()})
for x in res['pooled']:
    print(f"[{x['scope']:8s}] n={x['n']:4d} vs {x['baseline']:12s} diff={x['mean_diff']:+.4f} W/T/L={x['wins']}/{x['ties']}/{x['losses']} p={x['p']:.4f} holm={x['p_holm']:.4f}")
