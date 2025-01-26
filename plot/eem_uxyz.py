import numpy as np
import lhereader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def load(path):
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
    return np.array(momenta)

epem = []
for axis in 'uxyz':
    path = f'../run/epem_{axis}_0.0503rad/epem.lhe'
    print(path)
    momenta = load(path)
    P3, P4 = momenta.transpose(1, 0, 2)
    epem.append((P3, P4))
emem = []
for axis in 'uxyz':
    path = f'../run/emem_{axis}_0.0503rad/emem.lhe'
    print(path)
    momenta = load(path)
    P3, P4 = momenta.transpose(1, 0, 2)
    emem.append((P3, P4))

for epem_axis, (P7, P8) in zip('uxyz', epem):
    theta_7 = np.arctan2(np.hypot(P7[:,1], P7[:,2]), P7[:,3])
    for emem_axis, (P9, P10) in zip('uxyz', emem):
        theta_9 = np.arctan2(np.hypot(P9[:,1], P9[:,2]), P9[:,3])
        plt.hist2d(theta_7, theta_9, bins=100, density=True, norm=mcolors.LogNorm())
        plt.xlabel(r'$\theta_7$ [rad]')
        plt.ylabel(r'$\theta_9$ [rad]')
        cbar = plt.colorbar()
        cbar.set_label(f'{epem_axis}{emem_axis}')
        plt.tight_layout()
        plt.savefig(f'eem_uxyz_{epem_axis}{emem_axis}.pdf')
        plt.clf()
