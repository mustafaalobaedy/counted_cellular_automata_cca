"""Statistics for both confirmatory studies with the pretrained ResNet-18 probe added as a seventh method.
Original results are read (never written) from results/; the new baseline from
results/results_pretrained.jsonl. The tests are those of final_analysis.py / analysis_fruits.py:
Wilcoxon signed-rank (Pratt) paired by (task, seed), Holm-corrected within each family; Friedman over
(task, n) cells with average ranks, Nemenyi CD and Nemenyi p-values (Demsar, 2006).
Confirmatory study: CNN-aug = CNN+aug at n <= 30 and CNN-strong at n = 2000; seeds 10-14 at n = 2000,
as in final_analysis.py. Incomplete (task, n) cells are dropped and their number is reported.
Writes results/analysis_pretrained.json and figures/fig13_ranks_pretrained.png, fig14_accuracy_pretrained.png."""
import json, collections, numpy as np, matplotlib
from scipy.stats import wilcoxon, friedmanchisquare, rankdata, studentized_range
matplotlib.use('Agg'); import matplotlib.pyplot as plt

NEW = 'ResNet-18 probe'
STUDIES = {
 'confirm': dict(path='results/results_confirm.jsonl', sizes=[10, 30, 2000],
                 models=['CCA v2', 'LogReg', 'HOG+SVM', 'Random ferns', 'CNN', 'CNN-aug', NEW]),
 'fruits': dict(path='results/results_fruits.jsonl', sizes=[10, 30, 'all'],
                models=['CCA v2', 'LogReg', 'HOG+SVM', 'Random ferns', 'CNN', 'CNN+aug', NEW])}
NEED = {10: 10, 30: 10, 2000: 5, 'all': 5}

def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for i, j in enumerate(o): run = max(run, min(1.0, (len(p) - i) * p[j])); adj[j] = run
    return adj
def wil(d): return float(wilcoxon(d, zero_method='pratt').pvalue) if np.any(d != 0) else 1.0
mean = lambda d: float(np.mean(list(d.values())))

def load(study, cfg):
    acc = collections.defaultdict(dict); auc = collections.defaultdict(dict); role = {}
    def add(r, m):
        key = (r['task'], r['n'], m); role[r['task']] = r['role']
        if r['seed'] not in acc[key]: acc[key][r['seed']] = r['acc']; auc[key][r['seed']] = r['auc']
    for l in open(cfg['path']):
        r = json.loads(l); m = r['model']
        if study == 'confirm':
            if r['n'] == 2000 and m == 'CNN+aug': continue
            if m in ('CNN+aug', 'CNN-strong'): m = 'CNN-aug'
            if r['n'] == 2000 and r['seed'] > 14: continue
        add(r, m)
    for l in open('results/results_pretrained.jsonl'):
        r = json.loads(l)
        if r['study'] == study: add(r, r['model'])
    return acc, auc, role

def analyse(study, cfg):
    acc, auc, role = load(study, cfg); MODELS, SIZES = cfg['models'], cfg['sizes']
    tasks = list(dict.fromkeys(k[0] for k in acc))
    allcells = [(t, n) for n in SIZES for t in tasks]
    cells = [c for c in allcells if all(len(acc[(c[0], c[1], m)]) == NEED[c[1]] for m in MODELS)]
    res = dict(models=MODELS, complete_cells=len(cells), expected_cells=len(allcells), cells=[])
    for t, n in cells:
        res['cells'].append(dict(task=t, role=role[t], n=n, **{m: dict(acc=mean(acc[(t, n, m)]),
            sd=float(np.std(list(acc[(t, n, m)].values()), ddof=1)), auc=mean(auc[(t, n, m)])) for m in MODELS}))
    def diffs(cs, a, b): return np.array([acc[(t, n, a)][s] - acc[(t, n, b)][s] for t, n in cs for s in sorted(acc[(t, n, b)])])
    def test(d, **kw): return dict(**kw, pairs=len(d), mean_diff=float(d.mean()), wins=int((d > 0).sum()),
                                   ties=int((d == 0).sum()), losses=int((d < 0).sum()), p=wil(d))
    # CCA vs every baseline per size (one Holm family, as in the original analyses) and pooled over sizes
    block = [test(diffs([c for c in cells if c[1] == n], 'CCA v2', m), n=n, baseline=m) for n in SIZES for m in MODELS[1:]]
    for x, a in zip(block, holm([b['p'] for b in block])): x['p_holm'] = float(a)
    allp = [test(diffs(cells, 'CCA v2', m), baseline=m) for m in MODELS[1:]]
    for x, a in zip(allp, holm([x['p'] for x in allp])): x['p_holm'] = float(a)
    res['pooled'], res['pooled_all_sizes'] = block, allp
    # the new baseline against every other method, per size (own Holm family)
    vs = [test(diffs([c for c in cells if c[1] == n], NEW, m), n=n, other=m) for n in SIZES for m in MODELS[:-1]]
    for x, a in zip(vs, holm([x['p'] for x in vs])): x['p_holm'] = float(a)
    res['pretrained_vs_others'] = vs
    M = np.array([[mean(acc[(t, n, m)]) for m in MODELS] for t, n in cells])
    R = np.array([rankdata(-row) for row in M]); k, N = len(MODELS), len(cells)
    chi2, p = friedmanchisquare(*M.T); se = np.sqrt(k * (k + 1) / (6 * N))
    q = float(studentized_range.ppf(0.95, k, 1e6) / np.sqrt(2))
    res['friedman'] = dict(chi2=float(chi2), p=float(p), N=N, k=k, q_alpha=q, nemenyi_CD=float(q * se),
                           avg_rank=dict(zip(MODELS, map(float, R.mean(0)))),
                           **{f'avg_rank_n{n}': dict(zip(MODELS, map(float, R[[i for i, c in enumerate(cells) if c[1] == n]].mean(0))))
                              for n in SIZES if any(c[1] == n for c in cells)})
    res['nemenyi'] = [dict(baseline=m, rank_diff=float(R.mean(0)[i] - R.mean(0)[0]),
                           p=float(studentized_range.sf(abs(R.mean(0)[i] - R.mean(0)[0]) / se * np.sqrt(2), k, 1e6)))
                      for i, m in enumerate(MODELS) if i > 0]
    best = [[m for m, v in zip(MODELS, row) if v == row.max()] for row in M]
    res['best_counts'] = dict(collections.Counter(b[0] for b in best if len(b) == 1))
    res['tied_best_counts'] = dict(collections.Counter(m for b in best if len(b) > 1 for m in b))
    return res

out = {s: analyse(s, c) for s, c in STUDIES.items()}
from pretrained import ResNet18
net = ResNet18()
out['resnet18'] = dict(params_total=int(sum(p.numel() for p in net.parameters())),
                       params_backbone=int(sum(p.numel() for n_, p in net.named_parameters() if not n_.startswith('fc.'))))
secs = collections.defaultdict(list)
for l in open('results/results_pretrained.jsonl'):
    r = json.loads(l); secs[f"{r['study']}|{r['n']}"].append(r['sec'])
out['resnet18']['sec_median'] = {k: float(np.median(v)) for k, v in secs.items()}
json.dump(out, open('results/analysis_pretrained.json', 'w'), indent=1)

for s, res in [(k, v) for k, v in out.items() if k in STUDIES]:
    F = res['friedman']; MODELS = res['models']
    print(f"\n=== {s}: {res['complete_cells']}/{res['expected_cells']} complete cells | Friedman chi2={F['chi2']:.2f} p={F['p']:.2e} CD={F['nemenyi_CD']:.2f}")
    print('avg ranks', {m: round(v, 2) for m, v in sorted(F['avg_rank'].items(), key=lambda kv: kv[1])})
    for key in [k_ for k_ in F if k_.startswith('avg_rank_n')]:
        print(f'  {key}', {m: round(v, 2) for m, v in F[key].items()})
    print(f"{'task':28s} {'n':>4s} " + ' '.join(f'{m[:9]:>9s}' for m in MODELS))
    for c in res['cells']: print(f"{c['task']:28s} {str(c['n']):>4s} " + ' '.join(f"{c[m]['acc']:9.3f}" for m in MODELS))
    for x in res['pooled']:
        if x['baseline'] == NEW: print(f"n={str(x['n']):4s} CCA vs {NEW}: diff={x['mean_diff']:+.4f} W/T/L={x['wins']}/{x['ties']}/{x['losses']} p={x['p']:.2e} holm={x['p_holm']:.2e}")
    for x in res['pooled_all_sizes']:
        print(f"all sizes CCA vs {x['baseline']:15s} diff={x['mean_diff']:+.4f} holm={x['p_holm']:.2e}")
    print('sole best', res['best_counts'], '| tied best', res['tied_best_counts'])

# ---- figures ----
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
col = {'CCA v2': '#C0392B', 'LogReg': '#999999', 'HOG+SVM': '#27864A', 'Random ferns': '#B07D2B',
       'CNN': '#2E5C8A', 'CNN-aug': '#1B3A5C', 'CNN+aug': '#1B3A5C', NEW: '#7D3C98'}
title = {'confirm': 'MNIST / Fashion-MNIST', 'fruits': 'Fruits-360'}
LABEL = {'CCA v2': 'CCA', 'CNN-aug': 'CNN + aug', 'CNN+aug': 'CNN + aug', 'HOG+SVM': 'HOG + SVM'}
fig, axs = plt.subplots(1, 2, figsize=(11, 3.0))
for ax, (s, res) in zip(axs, [(k, out[k]) for k in STUDIES]):
    F = res['friedman']; ar = F['avg_rank']; order = sorted(ar, key=ar.get); y = np.arange(len(order))[::-1]
    ax.barh(y, [ar[m] for m in order], color=[col[m] for m in order], alpha=.85)
    for yy, m in zip(y, order): ax.text(ar[m] + 0.05, yy, f'{ar[m]:.2f}', va='center', fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([LABEL.get(m, m) for m in order]); ax.set_xlim(1, 7.4)
    ax.axvline(ar[order[0]] + F['nemenyi_CD'], color='k', ls=':', lw=1)
    ax.set_title(f"{title[s]}: {F['N']} cells, Friedman p = {F['p']:.1e}, CD = {F['nemenyi_CD']:.2f}", fontsize=8.5)
    ax.set_xlabel('average rank (1 = best)')
plt.tight_layout(); plt.savefig('figures/fig13_ranks_pretrained.png', dpi=200); plt.close()

fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, (s, res) in zip(axs, [(k, out[k]) for k in STUDIES]):
    MODELS, SIZES = res['models'], STUDIES[s]['sizes']
    for m in MODELS:
        v = [np.mean([c[m]['acc'] for c in res['cells'] if c['n'] == n]) for n in SIZES]
        ax.plot(range(len(SIZES)), v, marker='o', color=col[m], lw=2.4 if m in ('CCA v2', NEW) else 1.1,
                ls='--' if m in ('CNN-aug', 'CNN+aug') else '-', label=LABEL.get(m, m))
    ax.set_xticks(range(len(SIZES))); ax.set_xticklabels([str(n) for n in SIZES]); ax.grid(alpha=.3)
    ax.set_xlabel('training images per class'); ax.set_ylabel('mean test accuracy over tasks'); ax.set_title(title[s], fontsize=9)
axs[1].legend(fontsize=7.5, loc='lower right')
plt.tight_layout(); plt.savefig('figures/fig14_accuracy_pretrained.png', dpi=200); plt.close()
print('figures written')
