import numpy as np
import lhereader
import matplotlib.pyplot as plt
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
        assert particles[0].pdgid == 11 and particles[1].pdgid == 11
        particle_momenta = []
        for particle in particles:
            particle_momenta.append([particle.energy, particle.px, particle.py, particle.pz])
        momenta.append(particle_momenta)
    return np.array(momenta), xsec

data = np.load('../epem_example_1.0GeV_pT_0.00e+00GeV_eta_1.70e+00.npz')
points, weights = data['points'], data['weights']

pool = multiprocessing.Pool()

events = []
for axis in 'uxyzl':
    paths = []
    P3s, P4s, Ws = [], [], []
    for theta_3, W12 in zip(points, weights):
        path = f'../run/emem_{axis}_{theta_3:.4f}rad/emem.lhe'
        paths.append(path)
    for theta_3, W12, path, record in zip(points, weights, paths, pool.imap(load, paths)):
        print(path)
        momenta, W34 = record
        P3, P4 = momenta.transpose(1, 0, 2)
        print('%.4f' % theta_3, W12, W34)
        W = W12 * W34 * np.ones_like(P3[:,0])
        P3s.append(P3); P4s.append(P4); Ws.append(W)
    P3, P4, W = map(np.concatenate, (P3s, P4s, Ws))
    events.append((P3, P4, W))

for axis, (P3, P4, W) in zip('uxyzl', events):
    theta_9 = np.arctan2(np.hypot(P3[:,1], P3[:,2]), P3[:,3])
    cos_theta_9 = np.cos(theta_9)
    if axis == 'z': axis = 'r'
    plt.hist(cos_theta_9, bins=50, weights=W, density=True, histtype='step', label=f'{axis} polarized')
plt.xlabel(r'$\cos\theta^\prime_9$')
plt.ylabel(r'$\frac{1}{\sigma}\frac{\mathrm{d}\sigma}{\mathrm{d}\cos\theta^\prime_9}$')
plt.yscale('log')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('emem_uxyzl_theta.pdf')
plt.clf()

for axis, (P3, P4, W) in zip('uxyzl', events):
    phi_9 = np.arctan2(P3[:,2], P3[:,1])
    if axis == 'z': axis = 'r'
    plt.hist(phi_9, bins=10, weights=W, density=True, histtype='step', label=f'{axis} polarized')
plt.xlabel(r'$\phi_9$ [rad]')
plt.ylabel(r'$\frac{1}{\sigma}\frac{\mathrm{d}\sigma}{\mathrm{d}\phi_9}\ [\mathrm{rad}^{-1}]$')
plt.yscale('log')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('emem_uxyzl_phi.pdf')
plt.clf()
