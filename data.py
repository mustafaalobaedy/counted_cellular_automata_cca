"""Download MNIST and Fashion-MNIST (OpenML) and store as compressed NumPy arrays."""
import numpy as np
from sklearn.datasets import fetch_openml

def download():
    for name, out in (('mnist_784', 'mnist.npz'), ('Fashion-MNIST', 'fmnist.npz')):
        X, y = fetch_openml(name, version=1, return_X_y=True, as_frame=False)
        np.savez_compressed(out, X=X.astype(np.uint8), y=y.astype(int))

def load_pair(file, a, b, n_train, n_test, seed):
    """Binary task a (label 0) vs b (label 1): n_train and n_test images per class,
    sampled without overlap; pixel values scaled to [0, 1]."""
    d = np.load(file); X, y = d['X'], d['y']
    rng = np.random.default_rng(seed); out = []
    for cls in (a, b):
        idx = rng.permutation(np.where(y == cls)[0])
        out += [idx[:n_train], idx[n_train:n_train + n_test]]
    tr = np.concatenate([out[0], out[2]]); te = np.concatenate([out[1], out[3]])
    rng.shuffle(tr)
    f = lambda i: X[i].reshape(-1, 28, 28) / 255.0
    return f(tr), (y[tr] == b).astype(int), f(te), (y[te] == b).astype(int)

TASKS = {'MNIST 3v8': ('mnist.npz', 3, 8), 'MNIST 4v9': ('mnist.npz', 4, 9),
         'Fashion T-shirt v Shirt': ('fmnist.npz', 0, 6)}

if __name__ == '__main__':
    download()
