import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
import warnings

warnings.filterwarnings('ignore')

article_data = np.load("Article_experimental_data.npy")

n_jct = 80
lj = 1.42
cj = 5.388e-5
cg = 1.44e-7
ct = 2e-5

rg = 1e8
rjp = np.inf
rjs = 0
rt = np.inf

plot_nb = [2, 1]
model = "c"
fmin, fmax = 0, 1/np.sqrt(lj*cj)/2/np.pi

zs = lambda _w, _lj, _cj, _rjp, _rjs: 1/(1/(1j*_lj*_w) + 1j*_cj*_w + 1/_rjp) + _rjs
yg = lambda _w, _cg, _rg: 1j*_cg*_w + 1/_rg
yt = lambda _w, _ct, _rt: 1j*_ct*_w + 1/_rt

# Model definition
if model == "c":
    #Continuous
    gamma = lambda _w, _lj, _cj, _cg, _rjs, _rjp, _rg: -1j*np.sqrt(zs(_w, _lj, _cj, _rjp, _rjs)*yg(_w, _cg, _rg))
    zl = lambda _w, _lj, _cj, _cg, _rjs, _rjp, _rg: np.sqrt(zs(_w, _lj, _cj, _rjp, _rjs)/yg(_w, _cg, _rg))
elif model == "d":
    # Discrete no losses
    gamma = lambda _w, _lj, _cj, _cg, _rjs, _rjp, _rg: np.arccos(zs(_w, _lj, _cj, _rjp, _rjs)*yg(_w, _cg, _rg)/2+1)
    zl = lambda _w, _lj, _cj, _cg, _rjs, _rjp, _rg: np.sqrt(zs(_w, _lj, _cj, _rjp, _rjs)/yg(_w, _cg, _rg) *
                                                            np.exp(1j*gamma(_w, _lj, _cj, _cg, _rjs, _rjp, _rg)))
elif model == "dl":
    # Discrete losses
    gamma = lambda _w, _lj, _cj, _cg, _rjs, _rjp, _rg: -1j*np.arccosh(zs(_w, _lj, _cj, _rjp, _rjs)*yg(_w, _cg, _rg)/2+1)
    zl = lambda _w, _lj, _cj, _cg, _rjs, _rjp, _rg: np.sqrt(zs(_w, _lj, _cj, _rjp, _rjs)/yg(_w, _cg, _rg) *
                                                            np.exp(1j*gamma(_w, _lj, _cj, _cg, _rjs, _rjp, _rg)))
else:
    print('Model input should be either:\n"c"\n"d"\n"dl"')


def y_in_f(_ws):
    gam = gamma(_ws, lj, cj, cg, rjs, rjp, rg)
    zl_v = zl(_ws, lj, cj, cg, rjs, rjp, rg)
    yt_v = yt(_ws, ct, rt)
    v_in = np.cosh(1j * gam * n_jct) + zl_v * yt_v * np.sinh(1j * gam * n_jct)
    i_in = 2 * yt_v * np.cosh(1j * gam * n_jct) + (zl_v * yt_v ** 2 + 1 / zl_v) * np.sinh(1j * gam * n_jct)
    return i_in / v_in


def get_zeros_real(vals, threshold=1e-5):
    vals = np.array(vals)
    sgn_multi = np.sign(vals)
    eigen_ind_multi = [ind for ind, test in enumerate(sgn_multi[:-1] * sgn_multi[1:] == -1) if test]
    return [ws[ind] for ind in eigen_ind_multi if abs(vals[ind-1] - vals[ind])<threshold]


def y_in_newton(_im, _re):
    return y_in_f(_re+1j*_im)


ws = np.linspace(fmin, fmax, 1000000)*2*np.pi
y_in = y_in_f(ws)
re_y_in = np.real(y_in)
im_y_in = np.imag(y_in)
zeros_guesses = get_zeros_real(im_y_in)
print("Guesses :", [z/2/np.pi for z in zeros_guesses])

zeros = []
for z in zeros_guesses:
    try:
        sol = newton(y_in_f, z, maxiter=10000)
    except RuntimeError:
        continue
    if np.real(sol) > 1e-9:
        zeros.append(sol/2/np.pi)
print("Roots : ", zeros)

zeros_2 = []
for z in zeros_guesses:
    try:
        sol = z+1j*newton(y_in_newton,0, args=(z,), maxiter=10000)
    except RuntimeError:
        continue
    if np.real(sol) > 1e-9:
        zeros_2.append(sol/2/np.pi)
print("Roots 2 : ", zeros_2)

fig, axs = plt.subplots(2, len(plot_nb), figsize=(12, 8))
test_nb = len(plot_nb) == 1

if 0 in plot_nb:
    re_y_in = np.abs(re_y_in)
    im_y_in = np.abs(im_y_in)

    re_y_in = np.log(re_y_in)
    im_y_in = np.log(im_y_in)
    min_im, max_im = np.nanmin(im_y_in), np.nanmax(im_y_in)

    if test_nb:
        axs[0].plot(ws/2/np.pi, re_y_in)
        axs[1].plot(ws/2/np.pi, im_y_in)
        for z in zeros:
            axs[1].plot([np.real(z), np.real(z)], [min_im, max_im], "--+", color="green")
        for z in zeros_2:
            axs[1].plot([np.real(z), np.real(z)], [min_im, max_im], "--+", color="purple")
        for z in zeros_guesses:
            axs[1].plot([z/2/np.pi, z/2/np.pi], [min_im, max_im], "--+", color="red")
    else:
        axs[0, 0].plot(ws/2/np.pi, re_y_in)
        axs[1, 0].plot(ws/2/np.pi, im_y_in)
        for z in zeros:
            axs[1, 0].plot([np.real(z), np.real(z)], [min_im, max_im], "--+", color="green")
        for z in zeros_2:
            axs[1, 0].plot([np.real(z), np.real(z)], [min_im, max_im], "--+", color="purple")
        for z in zeros_guesses:
            axs[1, 0].plot([z/2/np.pi, z/2/np.pi], [min_im, max_im], "--+", color="red")

if 2 in plot_nb:
    re_ze = [np.real(z) for z in zeros]
    im_ze = [np.imag(z) for z in zeros]

    re_ze_2 = [np.real(z) for z in zeros_2]
    im_ze_2 = [np.imag(z) for z in zeros_2]

    if test_nb:
        axs[0].plot(re_ze, "+--")
        axs[0].plot(article_data[:len(re_ze)], "+--")
        axs[1].plot([r/2/i for r, i in zip(re_ze, im_ze)], "+--")

        axs[0].plot(re_ze_2, "+--")
        axs[0].plot(article_data[:len(re_ze_2)], "+--")
        axs[1].plot([r/2/i for r, i in zip(re_ze_2, im_ze_2)], "+--")
    else:
        axs[0, 0].plot(re_ze, "+--")
        axs[0, 0].plot(article_data[:len(re_ze)], "+--")
        axs[1, 0].plot([np.abs(r/2/i) for r, i in zip(re_ze, im_ze)], "+--")
        axs[1, 0].set_yscale("log")

        axs[0, 0].plot(re_ze_2, "+--")
        axs[0, 0].plot(article_data[:len(re_ze_2)], "+--")
        axs[1, 0].plot([np.abs(r/2/i) for r, i in zip(re_ze_2, im_ze_2)], "+--")

if 1 in plot_nb:
    try:
        zmin = min([np.imag(z) for z in zeros] + [np.imag(z) for z in zeros_2])
        zmax = max([np.imag(z) for z in zeros] + [np.imag(z) for z in zeros_2])
        zmin, zmax = 1.5*(zmin < 0)*zmin - zmax*(zmin > 0)/2, (1.5*(zmax > 0) + 2/3*(zmax < 0))*zmax
    except ValueError:
        zmin, zmax = -1, 1

    w_r = np.linspace(fmin, fmax, 1000)*2*np.pi
    w_i = np.linspace(zmin, zmax, 1000)*2*np.pi

    re, im = np.meshgrid(w_r, w_i)
    ws = re+1j*im

    y_in = y_in_f(ws)

    y_in_nrm = np.abs(y_in)
    y_in_ph = np.angle(y_in)

    y_in_nrm = np.log(y_in_nrm)

    if test_nb:
        axs[0].pcolor(re/2/np.pi, im/2/np.pi, y_in_nrm, cmap="hot_r")
        axs[1].pcolor(re/2/np.pi, im/2/np.pi, y_in_ph, cmap="hsv")
        for zr, zi in zip(np.real(zeros), np.imag(zeros)):
            axs[0].plot([zr], [zi], "--+", color="green", linewidth=2)
            axs[1].plot([zr], [zi], "--+", color="green", linewidth=2)
        for zr, zi in zip(np.real(zeros_2), np.imag(zeros_2)):
            axs[0].plot([zr], [zi], "--+", color="purple", linewidth=2)
            axs[1].plot([zr], [zi], "--+", color="purple", linewidth=2)
    else:
        axs[0, 1].pcolor(re/2/np.pi, im/2/np.pi, y_in_nrm, cmap="hot_r")
        axs[1, 1].pcolor(re/2/np.pi, im/2/np.pi, y_in_ph, cmap="hsv")
        for zr, zi in zip(np.real(zeros), np.imag(zeros)):
            axs[0, 1].plot([zr], [zi], "--+", color="green", linewidth=2)
            axs[1, 1].plot([zr], [zi], "--+", color="green", linewidth=2)
        for zr, zi in zip(np.real(zeros_2), np.imag(zeros_2)):
            axs[0, 1].plot([zr], [zi], "--+", color="purple", linewidth=2)
            axs[1, 1].plot([zr], [zi], "--+", color="purple", linewidth=2)

if test_nb:
    axs[0].grid()
    axs[1].grid()
else:
    axs[0, 0].grid()
    axs[0, 1].grid()
    axs[1, 0].grid()
    axs[1, 1].grid()
    axs[0, 1].set_ylim([zmin, zmax])
    axs[1, 1].set_ylim([zmin, zmax])
fig.tight_layout()
plt.show()
