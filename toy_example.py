"""Worked example: a tiny CCA run using the real cca.py functions.
Simplifications: one scale, one code family (cmp1), one rule stage, one region,
no count blur, no cross-fitting, no augmentation."""
import json, numpy as np
from cca import pix_codes, Stage0, RuleStage
from sklearn.linear_model import LogisticRegression

def vbar(col, top=0, bot=6, v=1.0):
    X = np.zeros((6, 6)); X[top:bot, col] = v; return X
def hbar(row, left=0, right=6, v=1.0):
    X = np.zeros((6, 6)); X[row, left:right] = v; return X
# training set: class 0 = vertical bar, class 1 = horizontal bar (4 images each)
Xtr = np.array([vbar(1, 0, 6, 1.0), vbar(2, 0, 5, 0.8), vbar(3, 1, 6, 0.9), vbar(2, 1, 6, 0.7),
                hbar(1, 0, 6, 1.0), hbar(2, 0, 5, 0.8), hbar(3, 1, 6, 0.9), hbar(4, 0, 6, 0.7)])
ytr = np.array([0, 0, 0, 0, 1, 1, 1, 1])
# test image: a short, slightly faint vertical bar
Xte = vbar(2, 1, 5, 0.9)[None]

s0 = Stage0(blur=0.0, alpha=0.5).fit(Xtr, ytr)
E0tr = s0(Xtr); E0 = s0(Xte)[0]
rule = RuleStage(grid=1, alpha=1.0).fit(E0tr, Xtr, ytr)
E1tr = rule(E0tr, Xtr) + E0tr; E1 = rule(E0[None], Xte)[0] + E0
Ftr = np.stack([E0tr.mean((1, 2)), E1tr.mean((1, 2))], 1)
lr = LogisticRegression(C=1.0).fit(Ftr, ytr)
S = np.array([E0.mean(), E1.mean()]); z = lr.intercept_[0] + lr.coef_[0] @ S; p = 1 / (1 + np.exp(-z))

codes = pix_codes(Xte)[0]
# chosen cell for the detailed walk-through: top end of the bar (row 1, col 2)
cy, cx = 1, 2
nb = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
P = np.pad(Xte[0], 1, mode='edge')
neigh = [float(P[1+cy+dy, 1+cx+dx]) for dy, dx in nb]
centre = float(Xte[0, cy, cx])
bits = [int(v > centre + 0.1) for v in neigh]; ink = int(centre > 0.3)
code = ink * 256 + sum(b << i for i, b in enumerate(bits))
assert code == codes[cy, cx]
# counts at this position and code
pos_codes = pix_codes(Xtr)[:, cy, cx]
n0 = int(((pos_codes == code) & (ytr == 0)).sum()); n1 = int(((pos_codes == code) & (ytr == 1)).sum())
N0, N1 = int((ytr == 0).sum()), int((ytr == 1).sum())
p0 = (n0 + 0.5) / (N0 + 512 * 0.5); p1 = (n1 + 0.5) / (N1 + 512 * 0.5)
L = float(np.log(p1) - np.log(p0)); assert abs(L - E0[cy, cx]) < 1e-9
# rule stage at a chosen cell
t = rule.t
q = lambda e: int(e > t) - int(e < -t)
# pick the rule-stage cell inside the test bar with the largest |R| and at least two non-zero digits
best = None
st_te = rule._state(E0[None], Xte)[0]
for yy in range(6):
    for xx in range(6):
        Pe_ = np.pad(E0, 1, mode='edge')
        dg = [q(E0[yy, xx]), q(Pe_[yy, 1+xx]), q(Pe_[1+yy, xx]), q(Pe_[1+yy, 2+xx]), q(Pe_[2+yy, 1+xx])]
        Rv = abs(rule.R[0, st_te[yy, xx]]); nz = sum(d != 0 for d in dg)
        margin = min(abs(abs(v) - t) for v in [E0[yy, xx], Pe_[yy, 1+xx], Pe_[1+yy, xx], Pe_[1+yy, 2+xx], Pe_[2+yy, 1+xx]])
        if nz >= 2 and margin > 0.05 and (best is None or Rv > best[0]): best = (Rv, yy, xx)
ry, rx = best[1], best[2]
Pe = np.pad(E0, 1, mode='edge')
digits_named = [('centre', E0[ry, rx]), ('up', Pe[1+ry-1, 1+rx]), ('left', Pe[1+ry, 1+rx-1]),
                ('right', Pe[1+ry, 1+rx+1]), ('down', Pe[1+ry+1, 1+rx])]
qd = [q(v) for _, v in digits_named]; d3 = [v + 1 for v in qd]
base3 = 0
for d in d3: base3 = base3 * 3 + d
ink_r = int(Xte[0, ry, rx] > 0.3); state = base3 * 2 + ink_r
assert state == rule._state(E0[None], Xte)[0, ry, rx]
# counts for this state in training evidence (one region)
st_tr = rule._state(E0tr, Xtr)
m0 = int((st_tr[ytr == 0] == state).sum()); m1 = int((st_tr[ytr == 1] == state).sum())
M0 = int((ytr == 0).sum() * 36); M1 = int((ytr == 1).sum() * 36)
r0 = (m0 + 1) / (M0 + 486); r1 = (m1 + 1) / (M1 + 486); R = float(np.log(r1) - np.log(r0))
assert abs(R - rule.R[0, state]) < 1e-9
out = dict(Xtr=Xtr.tolist(), ytr=ytr.tolist(), Xte=Xte[0].tolist(), codes=codes.tolist(),
           cell=[cy, cx], neigh=neigh, centre=centre, bits=bits, ink=ink, code=int(code),
           n0=n0, n1=n1, N0=N0, N1=N1, p0=p0, p1=p1, L=L, E0=E0.tolist(), t=float(t),
           tern=[[q(v) for v in row] for row in E0], rcell=[ry, rx],
           digits=[(nm, float(v), qq) for (nm, v), qq in zip(digits_named, qd)], base3=int(base3), ink_r=ink_r,
           state=int(state), m0=m0, m1=m1, M0=M0, M1=M1, r0=r0, r1=r1, R=R, E1=E1.tolist(),
           S0=float(S[0]), S1=float(S[1]), b=float(lr.intercept_[0]), w=lr.coef_[0].tolist(), z=float(z), p=float(p),
           Ftr=Ftr.tolist())
json.dump(out, open('results/toy.json', 'w'), indent=1)
np.set_printoptions(precision=2, suppress=True, linewidth=120)
print('codes\n', codes); print('E0\n', E0); print('t', t); print('tern\n', np.array(out['tern'])); print('E1\n', E1)
print('cell', cy, cx, 'neigh', neigh, 'centre', centre, 'bits', bits, 'code', code, 'n0 n1', n0, n1, 'L', round(L, 3))
print('rule cell', ry, rx, out['digits'], 'base3', base3, 'state', state, 'm0 m1', m0, m1, 'R', round(R, 3))
print('S', S, 'b', lr.intercept_, 'w', lr.coef_, 'z', z, 'p', p); print('train feats\n', Ftr)
