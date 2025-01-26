#!/usr/bin/env python3

import os
import numpy as np
import multiprocessing
import WZDCard
from get_p import get_p

pool = multiprocessing.Pool()
theta_ps = np.load('epem_example_1.0GeV_pT_0.00e+00GeV_eta_1.70e+00.npz')['points']
p_ps, p_es = np.array([get_p(theta_p) for theta_p in theta_ps]).transpose(1, 0, 2)
m_e = np.mean(np.sqrt(p_es[:,0]**2 - np.sum(p_es[:,1:]**2, axis=1)))
p_com_es = p_es + [[m_e, 0, 0, 0]]
e_com_es = np.sqrt(p_com_es[:,0]**2 - np.sum(p_com_es[:,1:]**2, axis=1))

for incoming_polarization in 'uxyzl':
    emem_card = WZDCard.WZDCard(f'cards/emem_{incoming_polarization}.sin')
    args = []
    for theta_p, e_com_e in zip(theta_ps, e_com_es):
        if os.path.exists(f'run/emem_{incoming_polarization}_{theta_p:.4f}rad'): continue
        args.append({
            'workdir': f'run/emem_{incoming_polarization}_{theta_p:.4f}rad',
            'nevent': 100000, 'com_energy': e_com_e,  # GeV
        })
    pool.map(emem_card.run, args)
