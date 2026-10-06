"""Tables and figures for the exploratory study, ablation, cost and evidence maps.
Reads results/results_explore.jsonl (from explore.py); writes results/analysis_explore.json and
figures/fig4_accuracy_vs_n.png, fig5_ablation.png, fig6_evidence_maps.png.
Mean +- standard deviation over seeds 0-2 (population SD, ddof = 0)."""
import json, time, collections, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from data import load_pair, TASKS
from run_experiments import MODELS
from cca2 import CCA2

R = collections.defaultdict(dict)
for l in open('results/results_explore.jsonl'):
    r = json.loads(l); R[(r['study'], r['task'], r['n'], r['model'])][r['seed']] = r
def stat(study, t, n, m):
    v = R[(study, t, n, m)]
    a = np.array([v[s]['acc'] for s in sorted(v)]); u = np.array([v[s]['auc'] for s in sorted(v)])
    return dict(acc=float(a.mean()), sd=float(a.std()), auc=float(u.mean()), seeds=len(a))
tasks = list(TASKS); res = {'main': [], 'ablation': [], 'dev': []}
for n in (10, 30, 2000):
    for t in tasks:
        res['main'].append(dict(task=t, n=n, **{m: stat('main', t, n, m) for m in MODELS if R[('main', t, n, m)]}))
for k in ('ablation', 'dev'):
    for (study, t, n, m) in sorted({x for x in R if x[0] == k}, key=lambda x: (x[2], x[3], tasks.index(x[1]))):
        res[k].append(dict(task=t, n=n, model=m, **stat(study, t, n, m)))

# ---- cost at n = 2000 (MNIST 3v8, seed 0): training and prediction time, stored numbers
Xtr, ytr, Xte, yte = load_pair('mnist.npz', 3, 8, 2000, 500, 0)
t0 = time.time(); cca = CCA2(seed=0).fit(Xtr, ytr); t1 = time.time(); cca.predict_proba(Xte); t2 = time.time()
tables = sum(c.s0.L.size + sum(r.R.size for r in c.rules) for cs in cca.chains for c in cs)
lr = cca.cal[-1]; fusion = lr.coef_.size + lr.intercept_.size
import baselines, torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from skimage.feature import hog
cost = {}
t0b = time.time(); m = LogisticRegression(max_iter=3000).fit(Xtr.reshape(len(Xtr), -1), ytr); t1b = time.time()
m.predict_proba(Xte.reshape(len(Xte), -1)); t2b = time.time()
cost['LogReg'] = dict(train=t1b - t0b, predict=t2b - t1b, params=str(m.coef_.size + m.intercept_.size))
H = lambda X: np.array([hog(x, orientations=8, pixels_per_cell=(7, 7), cells_per_block=(2, 2)) for x in X])
t0b = time.time(); m = SVC(probability=True, random_state=0).fit(H(Xtr), ytr); t1b = time.time()
m.predict_proba(H(Xte)); t2b = time.time()
cost['HOG+SVM'] = dict(train=t1b - t0b, predict=t2b - t1b, params=f'{m.support_vectors_.size:,} (support-vector values)')
torch.manual_seed(0); net = baselines.SmallCNN(); nparam = sum(p.numel() for p in net.parameters())
t0b = time.time(); baselines.run_cnn(Xtr, ytr, Xte[[0, -1]], yte[[0, -1]], 0, epochs=5); t1b = time.time()
net.eval()
with torch.no_grad():                      # prediction cost does not depend on the weights
    t2b = time.time(); torch.sigmoid(net(torch.tensor(Xte, dtype=torch.float32)[:, None])); t3b = time.time()
cost['CNN (5 epochs)'] = dict(train=t1b - t0b, predict=t3b - t2b, params=f'{nparam:,}')
cost['CCA v2'] = dict(train=t1 - t0, predict=t2 - t1, params=f'{tables:,} table entries + {fusion} fusion weights')
res['cost'] = cost

# ---- evidence maps of the cmp1 chain at 28x28 for one correctly classified test 3 and 8
chain = cca.chains[0][0]; p = cca.predict_proba(Xte)
maps = []
for cls in (0, 1):
    i = int(np.where((yte == cls) & ((p > 0.5) == yte))[0][0])
    E = chain.s0(Xte[i:i + 1]); Es = [E]
    for r in chain.rules: E = r(E, Xte[i:i + 1]) + E; Es.append(E)
    maps.append((i, Xte[i], [e[0] for e in Es]))
res['evidence_maps'] = [dict(test_index=i, label=['3', '8'][c], stage_means=[float(e.mean()) for e in Es])
                        for c, (i, _, Es) in enumerate(maps)]
json.dump(res, open('results/analysis_explore.json', 'w'), indent=1)

# ---- figures
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
sty = {'cca2': ('CCA v2', dict(c='#C0392B', lw=2.4, marker='o')), 'cnn': ('CNN', dict(c='#2E5C8A', lw=1.6, marker='s')),
       'cnn_aug': ('CNN + aug', dict(c='#2E5C8A', lw=1.6, marker='s', ls='--')),
       'hogsvm': ('HOG+SVM', dict(c='#27864A', lw=1.4, marker='^')), 'logreg': ('LogReg', dict(c='#888888', lw=1.2, marker='d'))}
fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, t in zip(axs, tasks):
    for m, (lab, st) in sty.items():
        cs = [c for c in res['main'] if c['task'] == t and m in c]
        ax.plot([[10, 30, 2000].index(c['n']) for c in cs], [c[m]['acc'] for c in cs], label=lab, **st)
    ax.set_xticks(range(3)); ax.set_xticklabels(['10', '30', '2000']); ax.grid(alpha=.3); ax.set_title(t, fontsize=10)
    ax.set_xlabel('training images per class')
axs[0].set_ylabel('test accuracy'); axs[2].legend(fontsize=8, loc='lower right')
plt.tight_layout(); plt.savefig('figures/fig4_accuracy_vs_n.png', dpi=200); plt.close()

order = [('ECA', 'ECA\n(stage 0 only)'), ('ECA + fixed CA rule', 'ECA + fixed CA rule'), ('CCA v1', 'CCA v1\n(learned rules)'),
         ('CCA v2 (no aug)', 'CCA v2\n(3 families)'), ('CCA v2', 'CCA v2 + shift aug')]
mean_abl = lambda m, n: float(np.mean([x['acc'] for x in res['ablation'] if x['model'] == m and x['n'] == n]))
res['ablation_mean'] = {f'{m}|{n}': mean_abl(m, n) for m, _ in order for n in (30, 2000)
                        if any(x['model'] == m and x['n'] == n for x in res['ablation'])}
json.dump(res, open('results/analysis_explore.json', 'w'), indent=1)
fig, ax = plt.subplots(figsize=(8.5, 3.4)); w = 0.36; allv = []
for i, (m, lab) in enumerate(order):
    # at n = 2000 'CCA v2' (auto settings, no augmentation) is the 3-family model
    for j, n in enumerate((30, 2000)):
        key = m if not (n == 2000 and m == 'CCA v2 (no aug)') else 'CCA v2'
        if n == 2000 and m == 'CCA v2': continue
        if f'{key}|{n}' not in res['ablation_mean']: continue
        v = res['ablation_mean'][f'{key}|{n}']; allv.append(v)
        ax.bar(i + (j - 0.5) * w, v, w, color=['#E59866', '#A93226'][j], label=f'{n} per class' if i == 0 else None)
        ax.text(i + (j - 0.5) * w, v + 0.001, f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)
ax.set_xticks(range(len(order))); ax.set_xticklabels([l for _, l in order], fontsize=8)
ax.set_ylim(np.floor(min(allv) * 50) / 50 - 0.02, max(allv) + 0.012); ax.set_ylabel('mean accuracy over 3 tasks')
ax.legend(loc='upper left'); ax.grid(axis='y', alpha=.3)
plt.tight_layout(); plt.savefig('figures/fig5_ablation.png', dpi=200); plt.close()

vmax = max(np.abs(e).max() for _, _, Es in maps for e in Es)
fig, axs = plt.subplots(2, 4, figsize=(7, 3.7), gridspec_kw=dict(width_ratios=[1.25, 1, 1, 1]))
names = ['E₀ (stage 0)', 'E₁ (rule stage 1)', 'E₂ (rule stage 2)']
for r, (i, x, Es) in enumerate(maps):
    axs[r, 0].imshow(x, cmap='gray_r'); axs[r, 0].set_ylabel(f'true class: {["3", "8"][r]}')
    for k, e in enumerate(Es):
        im = axs[r, k + 1].imshow(e, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axs[r, k + 1].set_xlabel(f'mean = {e.mean():+.3f}', fontsize=8)
        if r == 0: axs[r, k + 1].set_title(names[k], fontsize=9)
axs[0, 0].set_title('input', fontsize=9)
for ax in axs.ravel(): ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axs, fraction=0.03, label='log-odds evidence (red → "8", blue → "3")')
plt.savefig('figures/fig6_evidence_maps.png', dpi=200, bbox_inches='tight'); plt.close()
print(json.dumps({k: res[k] for k in ('ablation_mean', 'cost', 'evidence_maps')}, indent=1))
