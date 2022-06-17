import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

n_jct = 25
lj = 15/n_jct
cj = 1/lj/(15*2*np.pi)**2
cg = 1e-6
ct = 2/(lj*n_jct*(5*2*np.pi)**2)

rg_l = [1e8] # np.logspace(7, 8, 10)
rjp_l = [np.inf]
rjs_l = [0]
rt_l = [np.inf]

model = "c"
fmin, fmax = 0, 1/np.sqrt(lj*cj)/2/np.pi * 0.995

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


def lorentzian(_w, _wp, _width):
    return 1/(1+(2*(_w-_wp)/_width)**2)


def fit_lorentzian(_ws, _sqr_abs_z, _peaks, _properties, threshold=0.1, _n_point_fit=10 ** 3):
    _heights = _properties['peak_heights']
    _half_width_guess = _properties["right_ips"] - _properties["left_ips"]
    _list_guess_width = []
    _wps = []
    _qs = []
    for p, he, hwg in zip(_peaks, _heights, _half_width_guess):
        # ind = p
        # thresh_peak = he * threshold
        # while _sqr_abs_z[ind] > thresh_peak:
        #     ind -= 1
        #
        # _half_width = p-ind
        # w_peak = ws[p]
        # guess_width = ws[2*_half_width]

        w_peak = ws[p]
        guess_width = ws[int(hwg)] + (ws[1]-ws[0])*(hwg % 1)
        _list_guess_width.append(guess_width)

        # tight_ws = np.linspace(ws[p-_half_width], ws[p+_half_width], _n_point_fit)
        tight_ws = np.linspace(ws[p] - guess_width*2/3, ws[p] + guess_width*2/3, _n_point_fit)
        tight_z_in = z_in_f(tight_ws)
        tight_abs_z_in = np.abs(tight_z_in) ** 2
        tight_abs_z_in = tight_abs_z_in / np.nanmax(tight_abs_z_in)

        popt, pcov = curve_fit(lorentzian, tight_ws, tight_abs_z_in, p0=[w_peak, guess_width])
        [wp, width] = popt
        _wps.append(wp)
        _qs.append(abs(wp/width))
        print(f"guess : {wp/guess_width} | fit = {_qs[-1]}")

        if p == _peaks[0]:
            ax[1].plot(tight_ws / 2 / np.pi, tight_abs_z_in)
            ax[1].plot(tight_ws / 2 / np.pi, lorentzian(tight_ws, *popt))
        elif p == _peaks[1]:
            ax[2].plot(tight_ws / 2 / np.pi, tight_abs_z_in)
            ax[2].plot(tight_ws / 2 / np.pi, lorentzian(tight_ws, *popt))
    return _wps, _qs, _list_guess_width

variation = []
variation_params = np.array(np.meshgrid(rg_l, rjp_l, rjs_l, rt_l))[:, 0, :, 0, 0]
for rg, rjp, rjs, rt in zip(*variation_params):
    rg = int(rg)
    def z_in_f(_ws):
        gam = gamma(_ws, lj, cj, cg, rjs, rjp, rg)
        zl_v = zl(_ws, lj, cj, cg, rjs, rjp, rg)
        yt_v = yt(_ws, ct, rt)
        v_in = np.cosh(1j * gam * n_jct) + zl_v * yt_v * np.sinh(1j * gam * n_jct)
        i_in = 2 * yt_v * np.cosh(1j * gam * n_jct) + (zl_v * yt_v ** 2 + 1 / zl_v) * np.sinh(1j * gam * n_jct)
        return v_in / i_in

    ws = np.linspace(fmin, fmax, 1_000_000)*2*np.pi
    z_in = z_in_f(ws)
    sqr_abs_z = np.abs(z_in)**2
    log_sqr_abs = np.log(sqr_abs_z)/np.log(10)
    min_log_sqr_abs_z, max_sqr_abs_z = np.nanmin(log_sqr_abs), np.nanmax(log_sqr_abs)

    peaks, properties = find_peaks(sqr_abs_z, height=0, width=1)
    peaks_plot = ws[np.array(peaks)]

    print("Guesses :", [w / 2 / np.pi for w in peaks_plot])
    fig, ax = plt.subplots(3, 1, figsize=(8, 6))

    wps, qs, ind_half_widths = fit_lorentzian(ws, sqr_abs_z, peaks, properties)

    variation.append([wps, qs, ind_half_widths])

variation = np.array(variation)

# print("Guesses :", [w / 2 / np.pi for w in peaks_plot])
# fig, ax = plt.subplots(3, 1, figsize=(8, 6))

ax[0].plot(ws/2/np.pi, log_sqr_abs)

for w in peaks_plot:
    ax[0].plot([w/2/np.pi, w/2/np.pi], [min_log_sqr_abs_z, max_sqr_abs_z], "+--", color="red")


ax[0].set_xlabel("f (GHz)")
ax[0].set_ylabel(r"$2log10(|Y_in|)$")
ax[0].grid()
ax[1].grid()
ax[2].grid()

fig.tight_layout()
plt.show()
#
# fig, ax = plt.subplots(2, 1, figsize=(8, 6))
#
# ax[0].plot(wps, "+--")
# ax[1].plot(qs, "+--")
#
# ax[0].set_ylabel("f (GHz)")
# ax[0].set_ylabel("Q")
# ax[1].set_xlabel("Mode number")
# ax[1].set_yscale("log")
# ax[0].grid()
# ax[1].grid()
#
# fig.tight_layout()
# plt.show()

# fig, ax = plt.subplots(2, 1, figsize=(8, 6))
#
# ax[0].plot(variation[:, 0, 0], "+--")
# ax[1].plot(variation[:, 1, 0], "+--")
#
# ax[0].set_ylabel("f (GHz)")
# ax[1].set_ylabel("Q")
# ax[1].set_xlabel("Mode number")
# ax[1].set_yscale("log")
# ax[0].grid()
# ax[1].grid()
#
# fig.tight_layout()
# plt.show()