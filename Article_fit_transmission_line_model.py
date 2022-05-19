import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import warnings

warnings.filterwarnings('ignore')

article_data = np.load("Article_experimental_data.npy")
n_jct = 80

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
    #cj, cg, ct = abs(cj), abs(cg), abs(ct)
    ws = (np.linspace(0, 1/np.sqrt(lj*cj)/2/np.pi, 100000, dtype=complex)*2*np.pi)[1:]
    z_in_im_vals = np.imag(z_in(ws, _n_jct, lj, cj, cg, ct))
    sgn = np.sign(z_in_im_vals)
    f_roots = [ws[ind]/2/np.pi for ind, test in enumerate(sgn[:-1] * sgn[1:] == -1) if test]
    return f_roots[:9]

fig, ax = plt.subplots(figsize = (12,8))
[init_lj, init_cj, init_cg, init_ct] = [1.9, 4.06e-5, 1.093e-7, 1.445e-5]
line, = plt.plot(eigen_resonator([init_lj, init_cj, init_cg, init_ct], 80), "+--", label="Fit")
line_article, = plt.plot(article_data, "+--", label="Data")

ax.set_xlabel('Mode number')
ax.set_ylabel('Frequency [GHz]')
ax.legend()
plt.subplots_adjust(bottom=0.35)

axlj = plt.axes([0.25, 0.25, 0.65, 0.03])
lj_slider = Slider(
    ax=axlj,
    label='Junction inductance [nH]',
    valmin=1.8,
    valmax=2,
    valinit=init_lj,
)

axcj = plt.axes([0.25, 0.2, 0.65, 0.03])
cj_slider = Slider(
    ax=axcj,
    label='Junction capacity [nF]',
    valmin=1e-5,
    valmax=8e-5,
    valinit=init_cj,
)

axcg = plt.axes([0.25, 0.15, 0.65, 0.03])
cg_slider = Slider(
    ax=axcg,
    label='Capacity ground [nF]',
    valmin=1e-8,
    valmax=3e-7,
    valinit=init_cg,
)

axct = plt.axes([0.25, 0.1, 0.65, 0.03])
ct_slider = Slider(
    ax=axct,
    label='Resonator capacity [nF]',
    valmin=1e-6,
    valmax=5e-5,
    valinit=init_ct,
)


def update(val):
    line.set_ydata(eigen_resonator([lj_slider.val, cj_slider.val, cg_slider.val, ct_slider.val], n_jct))
    fig.canvas.draw_idle()


lj_slider.on_changed(update)
cj_slider.on_changed(update)
cg_slider.on_changed(update)
ct_slider.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    lj_slider.reset()
    cj_slider.reset()
    cg_slider.reset()
    ct_slider.reset()

button.on_clicked(reset)

plt.show()
