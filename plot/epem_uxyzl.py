import numpy as np
import lhereader
import matplotlib.pyplot as plt

def load(path):
    reader = lhereader.LHEReader(path)
    momenta = []
    for event in reader:
        particles = event.particles
        particles = list(filter(lambda x: x.status == 1, particles))
        assert len(particles) == 2
        particles = sorted(particles, key=lambda x: x.pdgid)
        assert particles[0].pdgid == -11 and particles[1].pdgid == 11
        particle_momenta = []
        for particle in particles:
            particle_momenta.append([particle.energy, particle.px, particle.py, particle.pz])
        momenta.append(particle_momenta)
    return np.array(momenta)

events = []
for axis in 'uxyzl':
    path = f'../run/epem_{axis}_0.0503rad/epem.lhe'
    print(path)
    momenta = load(path)
    P3, P4 = momenta.transpose(1, 0, 2)
    events.append((P3, P4))

for axis, (P3, P4) in zip('uxyzl', events):
    theta_7 = np.arctan2(np.hypot(P3[:,1], P3[:,2]), P3[:,3])
    if axis == 'z': axis = 'r'
    plt.hist(theta_7, bins=50, histtype='step', label=f'{axis} polarized')
plt.xlabel(r'$\theta^\prime_7$ [rad]')
plt.ylabel('Events')
plt.yscale('log')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('epem_uxyzl_theta.pdf')
plt.clf()

for axis, (P3, P4) in zip('uxyzl', events):
    phi_7 = np.arctan2(P3[:,2], P3[:,1])
    if axis == 'z': axis = 'r'
    plt.hist(phi_7, bins=10, histtype='step', label=f'{axis} polarized')
plt.xlabel(r'$\phi_7$ [rad]')
plt.ylabel('Events')
plt.yscale('log')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('epem_uxyzl_phi.pdf')
plt.clf()
