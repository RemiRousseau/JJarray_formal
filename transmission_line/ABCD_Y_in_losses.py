import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
import warnings

import analytic_continuation

warnings.filterwarnings('ignore')

article_data = np.load("../Article_experimental_data.npy")

n_jct = 20
lj = 15/n_jct
cj = 1/lj/(15*2*np.pi)**2
cg = 1e-6
ct = 2/(lj*n_jct*(5*2*np.pi)**2)

rg = 1e9
rjp = np.inf
rjs = 0
rt = np.inf

plot_nb = [0]
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


def get_zeros_real(vals, threshold=1e-4):
    vals = np.array(vals)
    sgn_multi = np.sign(vals)
    eigen_ind_multi = [ind for ind, test in enumerate(sgn_multi[:-1] * sgn_multi[1:] == -1) if test]
    return [ws[ind] for ind in eigen_ind_multi if abs(vals[ind-1] - vals[ind])<threshold]


def y_in_newton(_im, _re):
    return y_in_f(_re+1j*_im)

max_fit = 14.9

ws = np.linspace(fmin, fmax, 1_000_000)*2*np.pi
y_in = y_in_f(ws)
re_y_in = np.real(y_in)
im_y_in = np.imag(y_in)

zeros_guesses = get_zeros_real(im_y_in)
print("Guesses :", [z/2/np.pi for z in zeros_guesses])

ws_fit = (np.linspace(0, max_fit, 10_000)*2*np.pi)[1:]
a, b = analytic_continuation.rational_coefficients(ws_fit, y_in_f(ws_fit), reduce_orders=True)
fit = analytic_continuation.y_in(a, b, ws)
re_y_in_fit = np.real(fit)
im_y_in_fit = np.imag(fit)

print(f"Error fit : {sum([np.abs(tr-fi)**2 for tr, fi in zip(y_in, fit) if not np.isnan(tr-fi)])/len(y_in)}")

zeros = []
for z in zeros_guesses:
    try:
        sol = newton(y_in_f, z, maxiter=10000)
    except RuntimeError:
        print(f"Root {z/2/np.pi} failed to converge")
        continue
    if np.real(sol) > 1e-9:
        zeros.append(sol/2/np.pi)
print("Roots Newton : ", zeros)

zeros_2 = []
for z in zeros_guesses:
    try:
        sol = z+1j*newton(y_in_newton, 0, args=(z,), maxiter=10000)
    except RuntimeError:
        print(f"C Root {z/2/np.pi} failed to converge")
        continue
    if np.real(sol) > 1e-9:
        zeros_2.append(sol/2/np.pi)
print("Roots complex Newton : ", zeros_2)

zeros_fit = np.polynomial.polynomial.polyroots(a)
zeros_fit = [el/2/np.pi for el in zeros_fit if 0 < np.real(el) < max_fit*2*np.pi]
print(f"Roots fit : {zeros_fit}")

fig, axs = plt.subplots(2, len(plot_nb), figsize=(8, 6))
test_nb = len(plot_nb) == 1

if 0 in plot_nb:
    re_y_in = np.abs(re_y_in)
    im_y_in = np.abs(1/y_in)**2

    re_y_in = np.log(re_y_in)
    im_y_in = np.log(im_y_in)

    re_y_in_fit = np.abs(re_y_in_fit)
    im_y_in_fit = np.abs(im_y_in_fit)

    re_y_in_fit = np.log(re_y_in_fit)
    im_y_in_fit = np.log(im_y_in_fit)

    min_im, max_im = np.nanmin(im_y_in), np.nanmax(im_y_in)
    if test_nb:
        ax0, ax1 = axs[0], axs[1]
    else:
        ax0, ax1 = axs[0, 0], axs[1, 0]

    ax0.plot(ws/2/np.pi, re_y_in)
    ax1.plot(ws/2/np.pi, im_y_in)

    #ax0.plot(ws / 2 / np.pi, re_y_in_fit)
    #ax1.plot(ws / 2 / np.pi, im_y_in_fit)

    for z in zeros:
        ax1.plot([np.real(z), np.real(z)], [min_im, max_im], "--+", color="green")
    for z in zeros_2:
        ax1.plot([np.real(z), np.real(z)], [min_im, max_im], "--x", color="purple")
    for z in zeros_fit:
        ax1.plot([np.real(z), np.real(z)], [min_im, max_im], "--v", color="blue")
    for z in zeros_guesses:
        ax1.plot([z/2/np.pi, z/2/np.pi], [min_im, max_im], "--o", color="red")
    ax1.set_xlabel("frequency")
    ax0.set_ylabel(r"$log(|Re(Y_{in}|)$")
    ax1.set_ylabel(r"$log(|Im(Y_{in})|)$")

if 2 in plot_nb:
    re_ze = [np.real(z) for z in zeros]
    im_ze = [np.imag(z) for z in zeros]

    re_ze_2 = [np.real(z) for z in zeros_2]
    im_ze_2 = [np.imag(z) for z in zeros_2]

    re_ze_fit = [np.real(z) for z in zeros_fit]
    im_ze_fit = [np.imag(z) for z in zeros_fit]

    if test_nb:
        ax0, ax1 = axs[0], axs[1]
    else:
        ax0, ax1 = axs[0, 0], axs[1, 0]

    ax0.plot(re_ze, "+--", label="Newton")
    ax0.plot(re_ze_2, "x--", label="C Newton")
    ax0.plot(re_ze_fit, "v--", label="Rational fit")
    #ax0.plot(article_data[:len(zeros_guesses)], "o--", label="Article")
    ax0.plot([el/2/np.pi for el in zeros_guesses], "v--", label="Guesses")

    ax1.plot([r/2/i for r, i in zip(re_ze, np.abs(im_ze))], "+--", label="Newton")
    ax1.plot([r/2/i for r, i in zip(re_ze_2, np.abs(im_ze_2))], "x--", label="C Newton")
    ax1.plot([r/2/i for r, i in zip(re_ze_fit, np.abs(im_ze_fit))], "v--", label="Rational fit")

    ax1.set_xlabel("Mode number")
    ax0.set_ylabel("Frequency")
    ax1.set_ylabel("Q")
    ax0.legend()
    ax1.legend()

if 1 in plot_nb:
    try:
        zmin = min([np.imag(z) for z in zeros + zeros_2 + zeros_fit])
        zmax = max([np.imag(z) for z in zeros + zeros_2 + zeros_fit])
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
        ax0, ax1 = axs[0], axs[1]
    else:
        ax0, ax1 = axs[0, 1], axs[1, 1]

    ax0.pcolor(re/2/np.pi, im/2/np.pi, y_in_nrm, cmap="hot_r")
    ax1.pcolor(re/2/np.pi, im/2/np.pi, y_in_ph, cmap="hsv")
    ax0.plot(np.real(zeros), np.imag(zeros), "+", color="green", markersize=8, label="Newton")
    ax1.plot(np.real(zeros), np.imag(zeros), "+", color="green", markersize=8, label="Newton")
    ax0.plot(np.real(zeros_2), np.imag(zeros_2), "^", color="purple", markersize=8, label="C Newton")
    ax1.plot(np.real(zeros_2), np.imag(zeros_2), "^", color="purple", markersize=8, label="C Newton")
    ax0.plot(np.real(zeros_fit), np.imag(zeros_fit), "v", color="orange", markersize=8, label="Rational fit")
    ax1.plot(np.real(zeros_fit), np.imag(zeros_fit), "v", color="orange", markersize=8, label="Rational fit")


    def branch_cut(w):
        num = -rg*cg + np.sqrt(cg**2*rg**2 - (lj*cj*w)**2 + lj*cj)
        return num/2/np.pi/lj/cj


    def branch_cut_2(w):
        num = -rg*cg - np.sqrt(cg**2*rg**2 - 4*lj*cj**2*w**2+4*cj)
        return num/4/np.pi/lj/cj


    def branch_cut_zl(i_w):
        return np.sqrt((-2*cg*cj*lj*rg*i_w**3 + lj*cj*i_w**2 - 1)/(lj*cj*(2*rg*cg*i_w-1)))/2/np.pi


    for ax in [ax0, ax1]:
        ax.plot(w_r/2/np.pi, branch_cut(w_r), label=r"Branch cut $\gamma$", color="white", linewidth=1)
        ax.plot(w_r / 2 / np.pi, branch_cut_2(w_r), color="white", linewidth=1)
        ax.plot(branch_cut_zl(w_i), w_i, label="Branch cut $Z_l$", color="black", linewidth=1)


    ax0.set_ylabel("Re(f)")
    ax0.set_ylabel("Im(f)")
    ax1.set_ylabel("Re(f)")
    ax1.set_ylabel("Im(f)")
    ax0.set_title("Norm of $Y_{in}$ heat map")
    ax1.set_title("Phase of $Y_{in}$ heat map")
    ax0.legend()
    ax1.legend()

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
    axs[0, 1].set_xlim([fmin, fmax])
    axs[1, 1].set_xlim([fmin, fmax])
fig.tight_layout()
plt.show()
#
# i_ws = np.linspace(-0.1, 0.1, 1_000_000)*2*np.pi
# for i, z in enumerate(zeros_guesses):
#     plt.plot(i_ws/2/np.pi, y_in_newton(i_ws, z), label=str(i))
# plt.legend()
# plt.grid()
# plt.show()
