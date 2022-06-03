import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import warnings

warnings.filterwarnings('ignore')

n_plot = 8

n_jct = 13
L_j_0 = 7.5
fp = 15
f_lumped = 4

[init_lj_tot, init_cj, init_cg, init_ct] = [L_j_0, 1/(L_j_0/n_jct*(fp*2*np.pi)**2),
                                            1.093e-7, 2/(L_j_0*(f_lumped*2*np.pi)**2)]

zs = lambda w, lj, cj: 1j*lj*w/(1-lj*cj*w**2)
yg = lambda w, cg: 1j*cg*w
zt = lambda w, ct: 1/(1j*ct*w)
gamma = lambda w, lj, cj, cg: np.arccos(zs(w, lj, cj)*yg(w, cg)/2+1)
zl = lambda w, lj, cj, cg: np.sqrt(zs(w, lj, cj)/yg(w, cg)*np.exp(1j*gamma(w, lj, cj, cg)))


def z_in(w, _n_jct, lj, cj, cg, ct):
    _zl = zl(w, lj, cj, cg)
    _gamma = gamma(w, lj, cj, cg)
    _zt = zt(w, ct)
    return _zl*(_zt+1j*_zl*np.tan(_gamma*_n_jct/2))/(_zl+1j*_zt*np.tan(_gamma*_n_jct/2))


def eigen_resonator(x, _n_jct):
    [lj, cj, cg, ct] = list(np.abs(x))
    ws = (np.linspace(0, 1/np.sqrt(lj*cj)/2/np.pi, 100000, dtype=complex)*2*np.pi)[1:]
    z_in_im_vals = np.imag(z_in(ws, _n_jct, lj, cj, cg, ct))
    sgn = np.sign(z_in_im_vals)
    f_roots = [ws[ind]/2/np.pi for ind, test in enumerate(sgn[:-1] * sgn[1:] == -1) if test]
    return f_roots[:n_plot]

fig, ax = plt.subplots(figsize = (12,8))
line, = plt.plot(eigen_resonator([init_lj_tot/n_jct, init_cj, init_cg, init_ct], n_jct), "+--", label="Fit")
line_lumped, = plt.plot([-0.1, 0.1],[f_lumped,f_lumped], label="Ideal expectation")

ax.set_xlabel('Mode number')
ax.set_ylabel('Frequency [GHz]')
ax.legend()
plt.grid()
plt.subplots_adjust(bottom=0.3)

axlj = plt.axes([0.25, 0.2, 0.65, 0.03])
lj_slider = Slider(
    ax=axlj,
    label='Junction inductance [nH]',
    valmin=5,
    valmax=15,
    valinit=init_lj_tot,
)

axcg = plt.axes([0.25, 0.15, 0.65, 0.03])
cg_slider = Slider(
    ax=axcg,
    label='Capacity ground [nF]',
    valmin=1e-15,
    valmax=1e-6,
    valinit=init_cg,
)

axct = plt.axes([0.25, 0.1, 0.65, 0.03])
ct_slider = Slider(
    ax=axct,
    label='Resonator capacity [nF]',
    valmin=1e-15,
    valmax=1e-3,
    valinit=init_ct,
)


def update(val):
    _lj_tot, _cg, _ct = lj_slider.val, cg_slider.val, ct_slider.val
    _cj = 1/(_lj_tot/n_jct*(fp*2*np.pi)**2)
    f_exp = 1/np.sqrt(_lj_tot*_ct/2)/2/np.pi
    eig = eigen_resonator([_lj_tot/n_jct, _cj, _cg, _ct], n_jct)

    min_y = min(eig + [f_exp])
    max_y = max(eig + [f_exp])

    ax.set_ylim([min_y-0.1, max_y+0.1])
    line_lumped.set_ydata([f_exp, f_exp])
    line.set_ydata(eig)
    fig.canvas.draw_idle()


lj_slider.on_changed(update)
cg_slider.on_changed(update)
ct_slider.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    lj_slider.reset()
    cg_slider.reset()
    ct_slider.reset()

button.on_clicked(reset)

plt.show()
