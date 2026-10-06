# Counted Cellular Automaton (CCA)

Code to reproduce the experiments of the paper on the **Counted Cellular Automaton (CCA)**, a binary image
classifier with no convolutions and no gradient training. Every component, including the cellular-automaton
update rules, is a lookup table learned by counting.

## Installation

```bash
pip install -r requirements.txt
```

All experiments run on CPU. To make sure a GPU is not used, set `CUDA_VISIBLE_DEVICES=` before each command.
Run every script from the repository root.

## Files

| File | Content |
|---|---|
| `cca.py` | Shared counting functions, learned rule stage, single-family CCA v1 (used in the ablation) |
| `cca2.py` | Final multi-family model `CCA2` (used with its default `'auto'` settings) |
| `baselines.py` | Logistic regression, HOG + SVM, small CNN, CNN-strong |
| `ferns.py` | Random ferns baseline |
| `data.py` | MNIST / Fashion-MNIST download (OpenML) and task sampling |
| `data_fruits.py` | Fruits-360 download (Hugging Face mirror `PedroSampaio/fruits-360`) and preprocessing |
| `run_experiments.py` | Exploratory study, one run per call |
| `confirm.py`, `run_strong.py` | Confirmatory study (MNIST / Fashion-MNIST) |
| `final_analysis.py`, `final_figs.py` | Statistics and figures of the confirmatory study |
| `confirm_fruits.py` | Real-world study (Fruits-360) |
| `analysis_fruits.py`, `fruits_samples.py` | Statistics and figures of the real-world study |
| `toy_example.py` | Worked example of the paper (one prediction traced by hand) |
| `results/` | Raw per-run results (`*.jsonl`) and the analysis outputs (`*.json`) reported in the paper |

## Reproducing the results

**1. Confirmatory study** (7 MNIST / Fashion-MNIST tasks, 10, 30 and 2000 training images per class, seeds 10-19)

```bash
python data.py              # downloads MNIST and Fashion-MNIST -> mnist.npz, fmnist.npz
python confirm.py           # -> results/results_confirm.jsonl
python run_strong.py        # CNN-strong at n = 2000 -> results/results_confirm.jsonl
python final_analysis.py    # -> results/final_analysis.json
python final_figs.py        # -> figures/fig7-9
```

**2. Real-world study** (7 Fruits-360 tasks, 10, 30 and all training images per class)

```bash
python data_fruits.py       # downloads Fruits-360 -> fruits_data/fruits360_28.npz
python confirm_fruits.py    # -> results/results_fruits.jsonl
python analysis_fruits.py   # -> results/analysis_fruits.json, figures/fig10-11
python fruits_samples.py    # -> figures/fig12
```

**3. Exploratory study** (example: CCA with 30 images per class on MNIST 3 vs 8, seeds 0-2)

```bash
python run_experiments.py 30 "MNIST 3v8" cca2     # models: cca2, logreg, hogsvm, cnn, cnn_aug
```

**4. Worked example**

```bash
python toy_example.py       # -> results/toy.json
```

### Notes

- The study runners append one JSON line per (task, n, seed, model) and skip combinations that are already
  present, so they can be interrupted and restarted. The `results/` folder already holds the runs reported in
  the paper. To recompute them from scratch, delete or rename `results/results_*.jsonl` first.
- To recompute only the statistics and figures from the provided raw results, run `final_analysis.py`,
  `final_figs.py` and `analysis_fruits.py` directly. No dataset download is needed for this step.
- The CCA design and all baseline settings were fixed before the confirmatory and real-world studies. No setting
  is tuned on those data.

## Usage of the model

```python
from cca2 import CCA2
# X: (n, 28, 28) float array in [0, 1], object bright on a dark background; y: 0/1 labels
model = CCA2(seed=0).fit(X_train, y_train)
p = model.predict_proba(X_test)          # probability of class 1
```

## Data

- MNIST and Fashion-MNIST: OpenML (`mnist_784`, `Fashion-MNIST`).
- Fruits-360 (Mureșan and Oltean, 2018): Hugging Face mirror `PedroSampaio/fruits-360`, official train/test split.

## License

MIT License. See `LICENSE`.
