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

L_val = 2e-4
Lp_val = 1e-4
Cp_val = 5

L = Dipole('L', L_val)
Lp = Dipole('Lp', Lp_val)
Cp = Dipole('Cp', Cp_val)

circuit = [[N, W, N],
           [Lp, _, L],
           [N, _, N],
           [Cp, _, W],
           [N, W, N]]

wp = 15*2*np.pi
C_val = 1/L_val/wp**2
C = Dipole('C', C_val)

circuit_2 = [[N, W, N, W, N],
           [Lp, _, L, _, C],
           [N, _, N, _, N],
           [Cp, _, W, _, W],
           [N, W, N, W, N]]


c = Circuit(circuit)
c2 = Circuit(circuit_2)

fig = plt.figure(figsize=(9, 6))
gs = gridspec.GridSpec(ncols=2, nrows=2)
gs.update(left=0.10, right=0.95, wspace=0.0, hspace=0.1, top=0.95, bottom=0.05)
ax_eom = fig.add_subplot(gs[0, :])

omegas = np.linspace(0 * 2 * np.pi, 10 * 2 * np.pi, 2001)

guesses = [4 * 2 * np.pi]


eig_omegas, eig_phizpfs = c.rep_AC.solve_EIG(guesses)

eig_omegas_2, eig_phizpfs_2 = c2.rep_AC.solve_EIG(guesses)

wm = np.real(eig_omegas[0])
phi_l = eig_phizpfs[0][1]*hbar/2/e

w2_real = np.real(eig_omegas_2[0])

Lp_cal = np.abs(hbar*wm/2*L_val**2/phi_l**2 - L_val)
Cp_cal = 1/(L_val+Lp_cal)/wm**2

w_prime = 1/np.sqrt(Lp_cal*Cp_cal)

print(f"\nReal w : {w2_real/2/np.pi}")
print(f"Plasma freq : {wp/2/np.pi}\n")

print(f"Lp_cal : {Lp_cal}")
print(f"Cp_cal : {Cp_cal}\n")

print(f"Error L (%) :{abs(Lp_cal-Lp_val)/Lp_val*100}")
print(f"Error C (%) :{abs(Cp_cal-Cp_val)/Cp_val*100}\n")

w1 = 1/np.sqrt(1/wm**2 + 1/wp**2)
w2 = w1*np.sqrt(2/(1+np.sqrt(1-4*w1**4/wp**2/w_prime**2)))

w2_approx = 1/np.sqrt(1/wm**2 + 1/wp**2 - wm**2/w_prime**2/wp**2 + wm**4/w_prime**2/wp**4)

w2_approx_first = 1/np.sqrt(1/wm**2 + 1/wp**2 - wm**2/w_prime**2/wp**2)

print(f"Calc w : {w2/2/np.pi}")
print(f"Error w (%) : {np.abs(w2-w2_real)/w2_real*100}\n")

print(f"Aprox w : {w2_approx/2/np.pi}")
print(f"Error w (%) : {np.abs(w2_approx-w2_real)/w2_real*100}\n")

print(f"Aprox max w : {w2_approx_first/2/np.pi}")
print(f"Error w (%) : {np.abs(w2_approx_first-w2_real)/w2_real*100}")


