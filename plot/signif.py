import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit

sig = np.load('eem_uzl_epem.npz')
bkg = np.load('eem_uzl_uu.npz')
assert np.abs(np.max(sig['x'] - bkg['x'])) < 1e-4
assert np.abs(np.max(sig['y'] - bkg['y'])) < 1e-4
x, y = sig['x'], sig['y']
x, y = map(lambda x: np.mean([x[:-1], x[1:]], axis=0), (x, y))
vsig = sig['v']
vbkg = bkg['v']
sig = sig['h']
bkg = bkg['h']
diff = (sig >= 1e-4).astype('float') - (bkg >= 1e-4).astype('float')

thr = np.empty(len(y), dtype='int')
jmin, jmax = len(y), -1
for j in range(len(y)):
    imax = -1
    for i in range(len(x)):
        if diff[i, j] > 0: imax = i
    thr[j] = imax
    if thr[j] >= 0: jmin = min(jmin, j); jmax = max(jmax, j)
print(thr)
print(jmin, jmax)

def ytox(y, a, b):
    return a * y**4 + b
thr_y = y[jmin:jmax + 1]
thr_x = x[thr][jmin:jmax + 1]
(a, b), cov = curve_fit(ytox, thr_y, thr_x, p0=(-1, 1))
X, Y = np.meshgrid(y, x); Z = diff.T
plt.imshow(Z, extent=(-1, 1, -1, 1))
thr_x = -1 * np.ones(len(y))
thr_x[jmin:jmax + 1] = ytox(thr_y, a, b)
thr_y = y
plt.plot(thr_x, thr_y, label='fitted edge')
plt.xlabel(r'$\cos\theta^\prime_7$')
plt.ylabel(r'$\cos\theta^\prime_9$')
cbar = plt.colorbar()
cbar.set_label('binarize(S) - binarize(B)')
plt.legend()
plt.tight_layout()
plt.savefig(f'eem_uzl_threshold.pdf')
plt.clf()

thr = -1 * np.ones(len(y), dtype='int')
thr[jmin:jmax + 1] = np.argmax(x.reshape(-1, 1) >= thr_x[jmin:jmax + 1].reshape(1, -1), axis=0)
print(thr)
s, b, vs, vb = 0, 0, 0, 0
for j, imax in enumerate(thr):
    s += sig[:imax + 1, j].sum()
    b += bkg[:imax + 1, j].sum()
    vs += vsig[:imax + 1, j].sum()
    vb += vbkg[:imax + 1, j].sum()
r = s / b
vr = r * np.sqrt(vs / (s*s) + vb / (b*b))
print(s)
print(b)
print(r)
print(vr)
#r = sig[:int(len(x) * 0.75), int(len(y) * 0.125):int(len(y) * 0.875)].sum()
#s -= np.maximum(0, b)
#sf = 1.4e2 / r  # 1 s
#print(sf)
#s *= sf
#b *= sf
#print(s)
#print(b)
#print(s / np.sqrt(b))
