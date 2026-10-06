"""CNN-strong baseline at full data (n=2000), seeds 10-14, all 7 tasks. Resumable."""
import json, os, time
from data import load_pair
from baselines import run_cnn_strong
TASKS = {'MNIST 3v8': ('mnist.npz', 3, 8, 'dev'), 'MNIST 4v9': ('mnist.npz', 4, 9, 'dev'),
 'Fashion T-shirt v Shirt': ('fmnist.npz', 0, 6, 'dev'), 'MNIST 7v9': ('mnist.npz', 7, 9, 'held-out'),
 'MNIST 5v8': ('mnist.npz', 5, 8, 'held-out'), 'Fashion Pullover v Coat': ('fmnist.npz', 2, 4, 'held-out'),
 'Fashion Sneaker v Ankle boot': ('fmnist.npz', 7, 9, 'held-out')}
OUT = 'results/results_confirm.jsonl'
done = {(json.loads(l)['task'], json.loads(l)['seed'], json.loads(l)['model'], json.loads(l)['n']) for l in open(OUT)}
for task, (f, a, b, role) in TASKS.items():
    for s in range(10, 15):
        if (task, s, 'CNN-strong', 2000) in done: continue
        data = load_pair(f, a, b, 2000, 500, s); t0 = time.time()
        r = run_cnn_strong(*data, s)
        rec = dict(task=task, role=role, n=2000, seed=s, model='CNN-strong', acc=float(r['acc']), auc=float(r['auc']),
                   sec=round(time.time() - t0, 2))
        with open(OUT, 'a') as fh: fh.write(json.dumps(rec) + '\n')
print('STRONG DONE', flush=True)
