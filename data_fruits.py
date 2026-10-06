"""Fruits-360 (Muresan & Oltean, 2018) as a real-world test of CCA.
Source: Hugging Face mirror PedroSampaio/fruits-360 (131 classes, 67,690 train / 22,688 test,
100x100 RGB, one object on a white background). The official train/test split is kept.

Preprocessing (identical for every model, fixed before any run):
  RGB -> grayscale (PIL 'L') -> 28x28 (area resampling) -> inverted, so the white background
  becomes 0 and the object becomes 'ink', as in MNIST. Scaled to [0, 1]."""
import io, json, numpy as np

URL = 'https://huggingface.co/api/datasets/PedroSampaio/fruits-360/parquet/default/{}/0.parquet'
RAW = 'fruits_data/fruits360_{}.parquet'
NPZ = 'fruits_data/fruits360_28.npz'

def prepare(size=28):
    import os, urllib.request, pyarrow.parquet as pq
    from PIL import Image
    out = {}
    for split in ('train', 'test'):
        if not os.path.exists(RAW.format(split)):
            os.makedirs('fruits_data', exist_ok=True); urllib.request.urlretrieve(URL.format(split), RAW.format(split))
        t = pq.read_table(RAW.format(split))
        imgs = t.column('image').to_pylist()
        X = np.stack([255 - np.asarray(Image.open(io.BytesIO(d['bytes'])).convert('L')
                                       .resize((size, size), Image.BOX)) for d in imgs]).astype(np.uint8)
        out['X' + split] = X; out['y' + split] = np.asarray(t.column('label').to_pylist())
    names = json.loads(pq.read_schema(RAW.format('train')).metadata[b'huggingface'])['info']['features']['label']['names']
    np.savez_compressed(NPZ, names=np.array(names), **out)

def load_pair(a, b, n_train, seed, n_test=None):
    """Binary task: class name a (label 0) vs b (label 1). n_train images per class drawn from the
    official train split ('all' = every available image, balanced to the smaller class); the test set is
    the official test split of both classes, balanced to the smaller class (fixed across seeds)."""
    d = np.load(NPZ); names = list(d['names']); ia, ib = names.index(a), names.index(b)
    rng = np.random.default_rng(seed); tr = []; te = []
    ka = min((d['ytrain'] == ia).sum(), (d['ytrain'] == ib).sum()) if n_train == 'all' else n_train
    kt = min((d['ytest'] == ia).sum(), (d['ytest'] == ib).sum()) if n_test is None else n_test
    for c in (ia, ib):
        tr.append(rng.permutation(np.where(d['ytrain'] == c)[0])[:ka])
        te.append(np.random.default_rng(0).permutation(np.where(d['ytest'] == c)[0])[:kt])   # same for all seeds
    tr = np.concatenate(tr); te = np.concatenate(te); rng.shuffle(tr)
    f = lambda X, i: X[i] / 255.0
    return f(d['Xtrain'], tr), (d['ytrain'][tr] == ib).astype(int), f(d['Xtest'], te), (d['ytest'][te] == ib).astype(int)

if __name__ == '__main__':
    prepare()
