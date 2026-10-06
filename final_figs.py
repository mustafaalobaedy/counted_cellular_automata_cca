import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
A = json.load(open('results/final_analysis.json'))
M = ['CCA v2', 'LogReg', 'HOG+SVM', 'Random ferns', 'CNN', 'CNN-aug']
lab = {'CNN-aug': 'CNN + aug (strong at n=2000)'}
sty = {'CCA v2': dict(c='#C0392B', lw=2.4, marker='o'), 'LogReg': dict(c='#999999', lw=1.1, marker='d'),
       'HOG+SVM': dict(c='#27864A', lw=1.5, marker='^'), 'Random ferns': dict(c='#B07D2B', lw=1.1, marker='v'),
       'CNN': dict(c='#2E5C8A', lw=1.3, marker='s'), 'CNN-aug': dict(c='#2E5C8A', lw=1.7, marker='s', ls='--')}
tasks = list(dict.fromkeys(c['task'] for c in A['cells']))
fig, axs = plt.subplots(2, 4, figsize=(12, 5.8)); axs = axs.ravel()
for ax, t in zip(axs, tasks):
    cs = sorted([c for c in A['cells'] if c['task'] == t], key=lambda c: c['n'])
    for m in M: ax.plot(range(3), [c[m]['acc'] for c in cs], label=lab.get(m, m), **sty[m])
    ax.set_xticks(range(3)); ax.set_xticklabels(['10', '30', '2000']); ax.grid(alpha=.3)
    ax.set_title(t + (' (held-out)' if cs[0]['role'] == 'held-out' else ''), fontsize=8.5)
axs[0].set_ylabel('test accuracy'); axs[4].set_ylabel('test accuracy')
for ax in axs[3:7]: ax.set_xlabel('training images per class')
axs[7].axis('off'); h, l = axs[0].get_legend_handles_labels(); axs[7].legend(h, l, loc='center', fontsize=8.5)
plt.tight_layout(); plt.savefig('figures/fig7_confirm_accuracy.png', dpi=200); plt.close()

# average ranks with Nemenyi CD
F = A['friedman']; ar = F['avg_rank']; order = sorted(M, key=lambda m: ar[m]); cd = F['nemenyi_CD']
fig, ax = plt.subplots(figsize=(8, 2.8))
y = np.arange(len(order))[::-1]
ax.barh(y, [ar[m] for m in order], color=[sty[m]['c'] for m in order], alpha=.85)
for yy, m in zip(y, order): ax.text(ar[m] + 0.05, yy, f"{ar[m]:.2f}", va='center', fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels([lab.get(m, m) for m in order]); ax.set_xlim(1, 6.3)
best = ar[order[0]]; ax.axvline(best + cd, color='k', ls=':', lw=1)
ax.text(best + cd + 0.05, y[-1] - 0.45, f'best + CD ({cd:.2f})', fontsize=7.5)
ax.set_xlabel(f"average rank over {F['N']} task × size cells (1 = best); Friedman p = {F['p']:.1e}")
plt.tight_layout(); plt.savefig('figures/fig8_avg_ranks.png', dpi=200); plt.close()

# pooled mean differences CCA - baseline
P = [x for x in A['pooled'] if x['scope'] == 'all']; B = M[1:]
fig, ax = plt.subplots(figsize=(8.5, 3.3)); w = 0.26
for i, n in enumerate([10, 30, 2000]):
    v = [next(x for x in P if x['n'] == n and x['baseline'] == b) for b in B]
    xs = np.arange(len(B)) + (i - 1) * w
    bars = ax.bar(xs, [100 * x['mean_diff'] for x in v], w, label=f'n = {n}', color=['#F5B7B1', '#E74C3C', '#922B21'][i])
    for xx, x in zip(xs, v):
        s = '***' if x['p_holm'] < .001 else '**' if x['p_holm'] < .01 else '*' if x['p_holm'] < .05 else ''
        if s: ax.text(xx, 100 * x['mean_diff'] + (0.15 if x['mean_diff'] > 0 else -0.45), s, ha='center', fontsize=8)
ax.axhline(0, color='k', lw=.8); ax.set_xticks(range(len(B))); ax.set_xticklabels([lab.get(b, b).replace(' (strong at n=2000)', '\n(strong at n=2000)') for b in B], fontsize=8)
ax.set_ylabel('CCA − baseline accuracy (points)'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=.3)
ax.set_title('Pooled paired differences (all 7 tasks); * Holm-corrected Wilcoxon p<0.05, ** <0.01, *** <0.001', fontsize=8.5)
plt.tight_layout(); plt.savefig('figures/fig9_pooled_diffs.png', dpi=200); plt.close()
print('ok')
