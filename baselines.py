"""Baselines: logistic regression on pixels, HOG + RBF-SVM, and a small CNN."""
import time, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from skimage.feature import hog

torch.set_num_threads(4)

def run_logreg(Xtr, ytr, Xte, yte, seed):
    m = LogisticRegression(max_iter=3000).fit(Xtr.reshape(len(Xtr), -1), ytr)
    p = m.predict_proba(Xte.reshape(len(Xte), -1))[:, 1]
    return dict(acc=((p > 0.5) == yte).mean(), auc=roc_auc_score(yte, p))

def run_hog_svm(Xtr, ytr, Xte, yte, seed):
    H = lambda X: np.array([hog(x, orientations=8, pixels_per_cell=(7, 7),
                                cells_per_block=(2, 2)) for x in X])
    m = SVC(probability=True, random_state=seed).fit(H(Xtr), ytr)
    p = m.predict_proba(H(Xte))[:, 1]
    return dict(acc=((p > 0.5) == yte).mean(), auc=roc_auc_score(yte, p))

class SmallCNN(nn.Module):
    """conv(16)-pool-conv(32)-pool-fc(64)-fc(1): 105,281 parameters"""
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                              nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                              nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(s, x): return s.net(x).squeeze(1)

def run_cnn(Xtr, ytr, Xte, yte, seed, epochs=5):
    """Adam, lr 1e-3, batch 64, BCE loss. 5 epochs at n=2000/class, 30 at n<=30/class."""
    torch.manual_seed(seed)
    m = SmallCNN(); opt = torch.optim.Adam(m.parameters(), 1e-3); lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)[:, None]; yt = torch.tensor(ytr, dtype=torch.float32)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            b = perm[i:i + 64]; opt.zero_grad(); lossf(m(Xt[b]), yt[b]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p = torch.sigmoid(m(torch.tensor(Xte, dtype=torch.float32)[:, None])).numpy()
    return dict(acc=((p > 0.5) == yte).mean(), auc=roc_auc_score(yte, p))

def run_cnn_strong(Xtr, ytr, Xte, yte, seed, epochs=10):
    """Stronger CNN baseline for the confirmatory study: same architecture, on-the-fly random
    shifts of up to +-2 px every batch, 10 epochs, cosine learning-rate decay from 2e-3."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    m = SmallCNN(); opt = torch.optim.Adam(m.parameters(), 2e-3); lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)[:, None]; yt = torch.tensor(ytr, dtype=torch.float32)
    steps = epochs * ((len(Xt) + 63) // 64); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), 64):
            b = perm[i:i + 64]; xb = Xt[b]
            dy, dx = torch.randint(-2, 3, (2,), generator=g).tolist()
            xb = torch.roll(xb, shifts=(dy, dx), dims=(2, 3))
            opt.zero_grad(); lossf(m(xb), yt[b]).backward(); opt.step(); sched.step()
    m.eval()
    with torch.no_grad():
        p = torch.sigmoid(m(torch.tensor(Xte, dtype=torch.float32)[:, None])).numpy()
    return dict(acc=((p > 0.5) == yte).mean(), auc=roc_auc_score(yte, p))
