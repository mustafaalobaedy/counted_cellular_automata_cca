"""Transfer-learning baseline: a frozen ImageNet-pretrained ResNet-18 as a feature extractor, followed by
logistic regression (a 'linear probe'). Written for the revision of the main CCA paper (Oct 2026).

Fixed before any run, identical for every task and training-set size, nothing tuned:
  - Network: ResNet-18 with the torchvision ImageNet-1k weights (IMAGENET1K_V1, resnet18-f37072fd.pth,
    SHA-256 f37072fd47e8...), loaded into a plain-PyTorch copy of the torchvision architecture (torchvision
    itself is not needed). Frozen, eval mode (batch-norm running statistics), no fine-tuning.
  - Input: the same 28x28 arrays in [0, 1] that every other model receives; replicated to 3 channels,
    bilinearly resized to 224x224, normalised with the ImageNet mean and standard deviation.
  - Features: the 512-d global-average-pooled output of the last residual stage.
  - Classifier: features standardised on the training set, then LogisticRegression(C=1.0, max_iter=5000).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

URL = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

class Block(nn.Module):
    def __init__(s, cin, cout, stride):
        super().__init__()
        s.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); s.bn1 = nn.BatchNorm2d(cout)
        s.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); s.bn2 = nn.BatchNorm2d(cout)
        s.downsample = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                        if stride != 1 or cin != cout else None)
    def forward(s, x):
        o = F.relu(s.bn1(s.conv1(x))); o = s.bn2(s.conv2(o))
        return F.relu(o + (x if s.downsample is None else s.downsample(x)))

class ResNet18(nn.Module):
    """Same module names as torchvision.models.resnet18, so its state dict loads unchanged."""
    def __init__(s):
        super().__init__()
        s.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False); s.bn1 = nn.BatchNorm2d(64)
        s.maxpool = nn.MaxPool2d(3, 2, 1)
        cfg = [(64, 64, 1), (64, 128, 2), (128, 256, 2), (256, 512, 2)]
        for i, (a, b, st) in enumerate(cfg, 1):
            setattr(s, f'layer{i}', nn.Sequential(Block(a, b, st), Block(b, b, 1)))
        s.fc = nn.Linear(512, 1000)
    def features(s, x):
        x = s.maxpool(F.relu(s.bn1(s.conv1(x))))
        x = s.layer4(s.layer3(s.layer2(s.layer1(x))))
        return x.mean((2, 3))

_NET = None
def _net():
    global _NET
    if _NET is None:
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        m = ResNet18(); m.load_state_dict(torch.hub.load_state_dict_from_url(URL, check_hash=True, map_location='cpu'))
        _NET = m.eval().to(dev)
    return _NET

@torch.no_grad()
def features(X, batch=256):
    net = _net(); dev = next(net.parameters()).device
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1); std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(X), batch):
        x = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=dev).unsqueeze(1)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False).repeat(1, 3, 1, 1)
        out.append(net.features((x - mean) / std).cpu())
    return torch.cat(out).numpy().astype(np.float64)

def run_resnet_probe(Xtr, ytr, Xte, yte, seed):
    Ftr, Fte = features(Xtr), features(Xte)
    sc = StandardScaler().fit(Ftr)
    m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Ftr), ytr)
    p = m.predict_proba(sc.transform(Fte))[:, 1]
    return dict(acc=float(((p > 0.5) == yte).mean()), auc=float(roc_auc_score(yte, p)))
