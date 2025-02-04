import numpy as np
import lhereader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xml.etree.ElementTree as ET
import multiprocessing

def load(path):
    xsec = float(ET.parse(path).getroot().find('init/xsecinfo').attrib['totxsec'])
    reader = lhereader.LHEReader(path)
    momenta = []
    for event in reader:
        particles = event.particles
        particles = list(filter(lambda x: x.status == 1, particles))
        assert len(particles) == 2
        particles = sorted(particles, key=lambda x: x.pdgid)
        assert abs(particles[0].pdgid) == 11 and particles[1].pdgid == 11
        particle_momenta = []
        for particle in particles:
            particle_momenta.append([particle.energy, particle.px, particle.py, particle.pz])
        momenta.append(particle_momenta)
    return np.array(momenta), xsec

data = np.load('../epem_example_1.0GeV_pT_0.00e+00GeV_eta_1.70e+00.npz')
points, weights = data['points'], data['weights']

pool = multiprocessing.Pool()

epem_events = []
for axis in 'uzl':
    paths = []
    P3s, P4s, Ws = [], [], []
    for theta_3, W12 in zip(points, weights):
        path = f'../run/epem_{axis}10_{theta_3:.4f}rad/epem.lhe'
        paths.append(path)
    for theta_3, W12, path, record in zip(points, weights, paths, pool.imap(load, paths)):
        print(path)
        momenta, W34 = record
        P3, P4 = momenta.transpose(1, 0, 2)
        print('%.4f' % theta_3, W12, W34)
        W = W12 * W34 * np.ones_like(P3[:,0])
        P3s.append(P3); P4s.append(P4); Ws.append(W)
    P3, P4, W = map(np.concatenate, (P3s, P4s, Ws))
    epem_events.append((P3, P4, W))

emem_events = []
for axis in 'uzl':
    paths = []
    P3s, P4s, Ws = [], [], []
    for theta_3, W12 in zip(points, weights):
        path = f'../run/emem_{axis}10_{theta_3:.4f}rad/emem.lhe'
        paths.append(path)
    for theta_3, W12, path, record in zip(points, weights, paths, pool.imap(load, paths)):
        print(path)
        momenta, W34 = record
        P3, P4 = momenta.transpose(1, 0, 2)
        print('%.4f' % theta_3, W12, W34)
        W = W34 * np.ones_like(P3[:,0])  # no W12
        P3s.append(P3); P4s.append(P4); Ws.append(W)
    P3, P4, W = map(np.concatenate, (P3s, P4s, Ws))
    emem_events.append((P3, P4, W))

plt.figure(figsize=(4, 3), dpi=300)
P7s, P8s, P9s, P10s, Ws = [], [], [], [], []
for epem_axis, (P7, P8, W3) in zip('uzl', epem_events):
    for emem_axis, (P9, P10, W4) in zip('uzl', emem_events):
        if epem_axis + emem_axis not in ['zz', 'll']: continue
        P7s.append(P7); P8s.append(P8); P9s.append(P9); P10s.append(P10); Ws.append(W3 * W4)
P7, P8, P9, P10, W = map(np.concatenate, (P7s, P8s, P9s, P10s, Ws))
indexs = np.arange(P7.shape[0])
np.random.shuffle(indexs)
indexs = indexs[:indexs.shape[0] // 2]
P7, P8, P9, P10, W = map(lambda x: x[indexs], (P7, P8, P9, P10, W))
cos_theta_7 = np.cos(np.arctan2(np.hypot(P7[:,1], P7[:,2]), P7[:,3]))
cos_theta_9 = np.cos(np.arctan2(np.hypot(P9[:,1], P9[:,2]), P9[:,3]))
h, x, y, _, = plt.hist2d(cos_theta_7, cos_theta_9, bins=100, weights=W, density=True, norm=mcolors.LogNorm())
plt.xlabel(r'$\cos\theta^\prime_7$')
plt.ylabel(r'$\cos\theta^\prime_9$')
cbar = plt.colorbar()
cbar.set_label(r'$\frac{1}{\sigma}\frac{\mathrm{d}^2\sigma}{\mathrm{d}\cos\theta^\prime_7\mathrm{d}\cos\theta^\prime_9}$' + f' epem')
plt.tight_layout()
plt.savefig(f'eem_uzl10_epem.pdf')
np.savez('eem_uzl10_epem.npz', h=h, x=x, y=y)
plt.clf()
