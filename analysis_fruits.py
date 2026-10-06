"""Statistics and figures for the Fruits-360 study (same tests as final_analysis.py).
Wilcoxon signed-rank (Pratt) paired by (task, seed), Holm-corrected; Friedman over (task, n) cells with
average ranks and Nemenyi CD (Demsar, 2006)."""
import json, collections, numpy as np, matplotlib
from scipy.stats import wilcoxon, friedmanchisquare, rankdata, studentized_range
matplotlib.use('Agg'); import matplotlib.pyplot as plt

MODELS = ['CCA v2', 'LogReg', 'HOG+SVM', 'Random ferns', 'CNN', 'CNN+aug']
SIZES = [10, 30, 'all']; NEED = {10: 10, 30: 10, 'all': 5}
acc = collections.defaultdict(dict); auc = collections.defaultdict(dict); secs = collections.defaultdict(list); npc = {}
for l in open('results/results_fruits.jsonl'):
    r = json.loads(l); key = (r['task'], r['n'], r['model'])
    if r['seed'] not in acc[key]:
        acc[key][r['seed']] = r['acc']; auc[key][r['seed']] = r['auc']; secs[(r['n'], r['model'])].append(r['sec'])
        npc[(r['task'], r['n'])] = (r['n_per_class'], r['n_test'])
tasks = list(dict.fromkeys(k[0] for k in acc))
cells = [(t, n) for n in SIZES for t in tasks if all(len(acc[(t, n, m)]) == NEED[n] for m in MODELS)]
mean = lambda d: float(np.mean(list(d.values())))
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for i, j in enumerate(o): run = max(run, min(1.0, (len(p) - i) * p[j])); adj[j] = run
    return adj
def wil(d): return float(wilcoxon(d, zero_method='pratt').pvalue) if np.any(d != 0) else 1.0

res = {'cells': [], 'pooled': [], 'percell': [], 'complete_cells': len(cells)}
for t, n in cells:
    res['cells'].append(dict(task=t, n=n, n_per_class=npc[(t, n)][0], n_test=npc[(t, n)][1],
        **{m: dict(acc=mean(acc[(t, n, m)]), sd=float(np.std(list(acc[(t, n, m)].values()), ddof=1)),
                   auc=mean(auc[(t, n, m)])) for m in MODELS}))
block = []
for n in SIZES:
    cs = [t for t, nn in cells if nn == n]
    for m in MODELS[1:]:
        d = np.array([acc[(t, n, 'CCA v2')][s] - acc[(t, n, m)][s] for t in cs for s in sorted(acc[(t, n, m)])])
        if len(d): block.append(dict(n=n, baseline=m, pairs=len(d), mean_diff=float(d.mean()), wins=int((d > 0).sum()),
                                     ties=int((d == 0).sum()), losses=int((d < 0).sum()), p=wil(d)))
for x, a in zip(block, holm([b['p'] for b in block])): x['p_holm'] = float(a)
res['pooled'] = block
pcs = []
for t, n in cells:
    for m in MODELS[1:]:
        d = np.array([acc[(t, n, 'CCA v2')][s] - acc[(t, n, m)][s] for s in sorted(acc[(t, n, m)])])
        pcs.append(dict(task=t, n=n, baseline=m, mean_diff=float(d.mean()), p=wil(d)))
for x, a in zip(pcs, holm([x['p'] for x in pcs])): x['p_holm'] = float(a)
res['percell'] = pcs
M = np.array([[mean(acc[(t, n, m)]) for m in MODELS] for t, n in cells])
R = np.array([rankdata(-row) for row in M]); k, N = len(MODELS), len(cells)
chi2, p = friedmanchisquare(*M.T)
res['friedman'] = dict(chi2=float(chi2), p=float(p), N=N, k=k, avg_rank=dict(zip(MODELS, map(float, R.mean(0)))),
                       nemenyi_CD=float(2.850 * np.sqrt(k * (k + 1) / (6 * N))),
                       **{f'avg_rank_n{n}': dict(zip(MODELS, map(float, R[[i for i, c in enumerate(cells) if c[1] == n]].mean(0))))
                          for n in SIZES if any(c[1] == n for c in cells)})
# Nemenyi post-hoc p-values, CCA vs each baseline: q = |rank diff| / sqrt(k(k+1)/6N) * sqrt(2) ~ studentized range
se = np.sqrt(k * (k + 1) / (6 * N))
res['nemenyi'] = [dict(baseline=m, rank_diff=float(R.mean(0)[i] - R.mean(0)[0]),
                       p=float(studentized_range.sf(abs(R.mean(0)[i] - R.mean(0)[0]) / se * np.sqrt(2), k, 1e6)))
                  for i, m in enumerate(MODELS) if i > 0]
# pooled over all sizes: CCA vs each baseline, paired by (task, n, seed), Holm over the 5 tests
allp = []
for m in MODELS[1:]:
    d = np.array([acc[(t, n, 'CCA v2')][s] - acc[(t, n, m)][s] for t, n in cells for s in sorted(acc[(t, n, m)])])
    allp.append(dict(baseline=m, pairs=len(d), mean_diff=float(d.mean()), wins=int((d > 0).sum()),
                     ties=int((d == 0).sum()), losses=int((d < 0).sum()), p=wil(d)))
for x, a in zip(allp, holm([x['p'] for x in allp])): x['p_holm'] = float(a)
res['pooled_all_sizes'] = allp
# best method per cell, with ties reported separately (argmax alone would credit ties to the first model, CCA)
best = [[m for m, v in zip(MODELS, row) if v == row.max()] for row in M]
res['best_counts'] = dict(collections.Counter(b[0] for b in best if len(b) == 1))
res['tied_best_counts'] = dict(collections.Counter(m for b in best if len(b) > 1 for m in b))
res['train_sec_median'] = {f'{n}|{m}': float(np.median(v)) for (n, m), v in secs.items()}
json.dump(res, open('results/analysis_fruits.json', 'w'), indent=1)

print('complete cells', len(cells), '| best counts', res['best_counts'])
print('Friedman', {k_: v for k_, v in res['friedman'].items() if k_ in ('chi2', 'p', 'N', 'nemenyi_CD')})
print('avg ranks', {m: round(v, 2) for m, v in res['friedman']['avg_rank'].items()})
for n in SIZES:
    if f'avg_rank_n{n}' in res['friedman']: print(f'  n={n}', {m: round(v, 2) for m, v in res['friedman'][f'avg_rank_n{n}'].items()})
print(f"\n{'task':28s} {'n':>4s} " + ' '.join(f'{m[:9]:>9s}' for m in MODELS))
for c in res['cells']: print(f"{c['task']:28s} {str(c['n']):>4s} " + ' '.join(f"{c[m]['acc']:9.3f}" for m in MODELS))
print()
for x in res['pooled']:
    print(f"n={str(x['n']):4s} CCA vs {x['baseline']:12s} diff={x['mean_diff']:+.4f} W/T/L={x['wins']}/{x['ties']}/{x['losses']} p={x['p']:.4f} holm={x['p_holm']:.4f}")
for x in res['pooled_all_sizes']:
    print(f"all sizes CCA vs {x['baseline']:12s} diff={x['mean_diff']:+.4f} W/T/L={x['wins']}/{x['ties']}/{x['losses']} p={x['p']:.2e} holm={x['p_holm']:.2e}")
for x in res['nemenyi']: print(f"Nemenyi CCA vs {x['baseline']:12s} rank diff={x['rank_diff']:+.2f} p={x['p']:.4f}")
print('sole best', res['best_counts'], '| tied best', res['tied_best_counts'])
print('median train sec', res['train_sec_median'])

# ---- figures ----
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
sty = {'CCA v2': dict(c='#C0392B', lw=2.4, marker='o'), 'LogReg': dict(c='#999999', lw=1.1, marker='d'),
       'HOG+SVM': dict(c='#27864A', lw=1.5, marker='^'), 'Random ferns': dict(c='#B07D2B', lw=1.1, marker='v'),
       'CNN': dict(c='#2E5C8A', lw=1.3, marker='s'), 'CNN+aug': dict(c='#2E5C8A', lw=1.7, marker='s', ls='--')}
fig, axs = plt.subplots(2, 4, figsize=(12, 5.8)); axs = axs.ravel()
for ax, t in zip(axs, tasks):
    cs = [c for n in SIZES for c in res['cells'] if c['task'] == t and c['n'] == n]
    for m in MODELS: ax.plot(range(len(cs)), [c[m]['acc'] for c in cs], label=m, **sty[m])
    ax.set_xticks(range(len(cs))); ax.set_xticklabels([str(c['n_per_class']) for c in cs]); ax.grid(alpha=.3)
    ax.set_title(t, fontsize=8.5)
axs[0].set_ylabel('test accuracy'); axs[4].set_ylabel('test accuracy')
for ax in axs[3:7]: ax.set_xlabel('training images per class')
axs[7].axis('off'); h, l = axs[0].get_legend_handles_labels(); axs[7].legend(h, l, loc='center', fontsize=8.5)
plt.tight_layout(); plt.savefig('figures/fig10_fruits_accuracy.png', dpi=200); plt.close()

F = res['friedman']; ar = F['avg_rank']; order = sorted(MODELS, key=lambda m: ar[m]); cd = F['nemenyi_CD']
fig, ax = plt.subplots(figsize=(8, 2.8)); y = np.arange(len(order))[::-1]
ax.barh(y, [ar[m] for m in order], color=[sty[m]['c'] for m in order], alpha=.85)
for yy, m in zip(y, order): ax.text(ar[m] + 0.05, yy, f"{ar[m]:.2f}", va='center', fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels(order); ax.set_xlim(1, 6.3)
ax.axvline(ar[order[0]] + cd, color='k', ls=':', lw=1)
ax.text(ar[order[0]] + cd + 0.05, y[-1] - 0.45, f'best + CD ({cd:.2f})', fontsize=7.5)
ax.set_xlabel(f"Fruits-360: average rank over {N} task × size cells (1 = best); Friedman p = {F['p']:.1e}")
plt.tight_layout(); plt.savefig('figures/fig11_fruits_ranks.png', dpi=200); plt.close()
print('figures written')
