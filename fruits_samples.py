"""Figure: example Fruits-360 images (original RGB and the 28x28 inverted-grayscale model input) for each task."""
import io, numpy as np, pyarrow.parquet as pq, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from confirm_fruits import TASKS
t = pq.read_table('fruits_data/fruits360_train.parquet'); y = np.asarray(t.column('label').to_pylist())
d = np.load('fruits_data/fruits360_28.npz'); names = list(d['names'])
fig, axs = plt.subplots(4, 7, figsize=(10, 6))
for j, (task, (a, b)) in enumerate(TASKS.items()):
    for k, c in enumerate((a, b)):
        i = np.where(y == names.index(c))[0][40]
        axs[2*k, j].imshow(Image.open(io.BytesIO(t.column('image')[int(i)].as_py()['bytes'])))
        axs[2*k+1, j].imshow(d['Xtrain'][i], cmap='gray', vmin=0, vmax=255)
        axs[2*k, j].set_title(c, fontsize=8)
for ax in axs.ravel(): ax.set_xticks([]); ax.set_yticks([])
for r, l in enumerate(['class 0 (RGB)', 'model input', 'class 1 (RGB)', 'model input']): axs[r, 0].set_ylabel(l, fontsize=8)
plt.tight_layout(); plt.savefig('figures/fig12_fruits_samples.png', dpi=150)
