# -*- coding: utf-8 -*-

from EOMcircuit.functions import Dipole, Node, Hole, Circuit, Representation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize, root
import scipy.constants
from numpy.linalg import eig

plt.close('all')

k = scipy.constants.k
e = scipy.constants.e
h = scipy.constants.h
hbar = scipy.constants.hbar
phi0 = hbar / 2 / e
sci = hbar / phi0 ** 2
conv_L = 2 * np.pi * hbar / (4 * e * np.pi) ** 2  # conversion factor from LJ nH to EJ GHz
# LJ = conv_L/EJ
conv_C = e ** 2 / 2 / (hbar * 2 * np.pi)  # conversion factor from C nF to EC GHz
# C = conv_c/EC

W = Dipole('W')

_ = None

N = Node('')  # expect this for dummy nodes
G = Node('G')
E = Node('E')

Cg_val = 1e-4
Ct_val = 1e-6
Lj_val = 10

Cg = Dipole('Cg', Cg_val)
Ct = Dipole('Ct', Ct_val)
Lj = Dipole('Lj', Lj_val)

# circuit = [[N, Lj, N, Lj, N, Lj, N, Lj, N, Lj, N, Lj, N, W, N],
#             [Ct, _, Cg, _, Cg, _, Cg, _, Cg, _, Cg, _, Cg, _, Ct],
#             [G, _, G, _, G, _, G, _, G, _, G, _, G, _, G]]

# circuit = [[N , Cg, W, W , N, Cg, N],
#             [W , _ , _, _ , W, _ , W],
#             [N , Lj, N, Lj, N, Lj, N],
#             [W , _ , W, _ , _, _ , W],
#             [N , Cg, N, Cg, W, W , N],
#             [Ct, _ , _, _ , _, _ , Ct],
#             [G , _ , _, _ , _, _ , G]]

circuit = [[N , Cg, N, Cg, N, W , N],
           [W , _ , W, _ , _, _ , W],
           [N , Lj, N, Lj, N, Lj, N],
           [W , _ , _, _ , W, _ , W],
           [N , W , N, Cg, N, Cg, N],
           [Ct, _ , _, _ , _, _ , Ct],
           [G , _ , _, _ , _, _ , G]]

fig_circuit, ax_circuit = plt.subplots()
c = Circuit(circuit)  # parse circuit and much more is done
c.plot(ax_circuit)
print('### Circuit plotted ###')

fig = plt.figure(figsize=(9, 6))
gs = gridspec.GridSpec(ncols=2, nrows=2)
gs.update(left=0.10, right=0.95, wspace=0.0, hspace=0.1, top=0.95, bottom=0.05)
ax_eom = fig.add_subplot(gs[0, :])

omegas = np.linspace(0 * 2 * np.pi, 10 * 2 * np.pi, 2001)

# Esum = ELa+ELb+EL
# mat = np.array([[8*ECa*ELa*(ELb+EL)/Esum, -8*ECa*ELa*ELb/Esum],
#                 [-8*ECb*ELb*ELa/Esum, 8*ECb*ELb*(ELa+EL)/Esum]])
# e, v = eig(mat)

guesses = [0.09 * 2 * np.pi, 2.6 * 2 * np.pi]
eig_omegas, eig_phizpfs = c.rep_AC.display_eom(ax_eom, omegas, guesses=guesses,
                                               log_kappa=True)  # , kappas=kappas, guesses=guesses)#kappas=kappas

print(eig_omegas / 2 / np.pi)

eig_omegas, eig_phizpfs = c.rep_AC.solve_EIG(guesses)

### plot modes

ax0 = fig.add_subplot(gs[1, :])
c.plot(ax0)
c.rep_AC.plot_phi(ax0, 3 * np.real(eig_phizpfs[0]), offset=0.3, color='C0')  # 4* -> magnification for plot only
c.rep_AC.plot_phi(ax0, 3 * np.real(eig_phizpfs[1]), offset=0.5, color='C1')  # 4* -> magnification for plot only

phi_a = np.real(eig_phizpfs[0][2])
phi_b = np.real(eig_phizpfs[1][2])
