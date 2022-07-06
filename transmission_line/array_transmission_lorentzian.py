import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, minimize
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore', category=RuntimeWarning)


def lorentzian(_w, _wp, _width):
    return 1 / (1 + (2 * (_w - _wp) / _width) ** 2)


class ArrayLorentzian:
    def __init__(self, model, fmin=0, fmax=None, n_point_init=10 ** 6, n_point_fit=5 * 10 ** 2,
                 threshold_refinement=0.2, guess_width_ratio=2 / 3, ratio_fmax_plasma=0.995, show_progression=True,
                 show_fit_window=False):

        self._hbar = 1.054571817e-34
        self._e = 1.602176634e-19
        self._E_J_GHz = None

        self._n_jct = None
        self._lj = None
        self._cj = None
        self._cg = None
        self._ct = None

        self._model = model

        self._fmin = None
        self._fmax = None
        self._ratio_fmax_plasma = ratio_fmax_plasma

        self._resistor_losses = True
        self._rjp = np.inf
        self._rjs = 0
        self._rg = np.inf
        self._rt = np.inf

        self._tandj = 0
        self._tandg = 0
        self._tandt = 0

        self._lossless = True

        self._rjp_l = np.inf
        self._rjs_l = 0
        self._rg_l = np.inf
        self._rt_l = None
        self._tandj_l = None
        self._tandg_l = None
        self._tandt_l = None
        self._variation_params = None

        self._n_point_init = n_point_init
        self._n_point_fit = n_point_fit

        self._ws_init = None
        self._z_in_init = None
        self._sqr_abs_z = None
        self._log_sqr_abs = None
        self._min_log_sqr_abs_z = None
        self._max_sqr_abs_z = None

        self._peaks_indices = None
        self._peaks_properties = None
        self._peak_plot = None
        self._n_mode = None

        self._popts = None
        self._pcovs = None

        self._list_guess_width = None
        self._wps = None
        self._qs = None

        self._v_points_modes = None
        self._v_diffs_modes = None
        self._phi_zpf = None
        self._perfect_ground_state_phi_zpf = None
        self._kerr = None

        self._sweeped = None

        self._variations_wps = None
        self._variations_qs = None

        self._guess_width_ratio = guess_width_ratio
        self._threshold_refinement = threshold_refinement

        self._show_progression = show_progression
        self._show_fit_window = show_fit_window

        self._zs = self._zs_resistor
        self._yg = self._yg_resistor
        self._yt = self._yt_resistor

        model_lst = model.split("_")
        if model_lst[0] == "ground":
            self._z_in_f = self._z_in_ground
            if model_lst[1] == "c":
                self._gamma = self._gamma_c
                self._zl = self._zl_c
            elif model_lst[1] == "d":
                self._gamma = self._gamma_d
                self._zl = self._zl_d
            elif model_lst[1] == "dl":
                self._gamma = self._gamma_dl
                self._zl = self._zl_dl
            else:
                raise NameError("In model variable, 'ground' should be followed by either 'c', 'd' or 'dl'.")
        elif model_lst[0] == "cable":
            if model_lst[1] == "c":
                self._z_in_f = self._z_in_cable
                self._gamma = lambda _ws: self._gamma_c(_ws)*np.sqrt(2)
            else:
                raise NameError("In model variable, 'cable' should be followed by 'c'.")
        elif model_lst[0] == "perfect":
            self._z_in_f = self._z_in_perfect
        else:
            raise NameError('Entry model should be either "ground_c", "ground_d", "ground_dl", "cable_c",' +
                            ' "cable_cp" or "perfect.')

    def init_params(self, n_jct, lj, cj, cg, ct):
        self._n_jct = n_jct
        self._lj = lj
        self._cj = cj
        self._cg = cg
        self._ct = ct

        self._fmin = 0
        self._fmax = 1 / np.sqrt(lj * cj) / 2 / np.pi * self._ratio_fmax_plasma

        self._E_J_GHz = self._hbar / 4 / self._e ** 2 / self._lj

    def init_losses_resistor(self, rjp, rjs, rg, rt):
        self._rjp = rjp
        self._rjs = rjs
        self._rg = rg
        self._rt = rt
        self._resistor_losses = True
        self._zs = self._zs_resistor
        self._yg = self._yg_resistor
        self._yt = self._yt_resistor
        if not (self._rjp == np.inf and self._rjs == 0 and self._rg == np.inf and self._rt == np.inf):
            self._lossless = False
        else:
            self._lossless = True

    def init_losses_tand(self, tandj, tandg, tandt):
        self._tandj = tandj
        self._tandg = tandg
        self._tandt = tandt
        self._resistor_losses = False
        self._zs = self._zs_tan
        self._yg = self._yg_tan
        self._yt = self._yt_tan
        if not (self._tandj == 0 and self._tandg == 0 and self._tandt == 0):
            self._lossless = False
        else:
            self._lossless = True

    def set_fmin_max(self, fmin, fmax):
        self._fmin, self._fmax = fmin, fmax

    def _zs_resistor(self, _w):
        try:
            res = 1 / (1 / (1j * self._lj * _w) + 1 / (1 / (1j * self._cj * _w) + self._rjs) + 1 / self._rjp)
        except ZeroDivisionError:
            res = np.nan
        return res

    def _zs_tan(self, _w):
        try:
            res = 1 / (1 / (1j * self._lj * _w) + (1j + self._tandj) * self._cj * _w)
        except ZeroDivisionError:
            res = np.nan
        return res

    def _yg_resistor(self, _w):
        return 1j * self._cg * _w + 1 / self._rg

    def _yg_tan(self, _w):
        return (1j + self._tandg) * self._cg * _w

    def _yt_resistor(self, _w):
        return 1j * self._ct * _w + 1 / self._rt

    def _yt_tan(self, _w):
        return (1j + self._tandt) * self._ct * _w

    # Model definitions:
    # Continuous
    def _gamma_c(self, _w):
        try:
            res = 1j * np.sqrt(self._zs(_w) * self._yg(_w))
        except ZeroDivisionError:
            res = np.nan
        return res

    def _zl_c(self, _w):
        try:
            res = np.sqrt(self._zs(_w) / self._yg(_w))
        except ZeroDivisionError:
            res = np.nan
        return res

    # Discrete
    def _gamma_d(self, _w):
        return -np.arccos(self._zs(_w) * self._yg(_w) / 2 + 1)

    def _zl_d(self, _w):
        try:
            res = np.sqrt(self._zs(_w) / self._yg(_w) * np.exp(1j * self._gamma_d(_w)))
        except ZeroDivisionError:
            res = np.nan
        return res

    # Discrete for losses
    def _gamma_dl(self, _w):
        return 1j * np.arccosh(self._zs(_w) * self._yg(_w) / 2 + 1)

    def _zl_dl(self, _w):
        try:
            res = np.sqrt(self._zs(_w) / self._yg(_w) * np.exp(1j * self._gamma_dl(_w)))
        except ZeroDivisionError:
            res = np.nan
        return res

    def _z_in_ground(self, _ws):
        gam = self._gamma(_ws)
        zl_v = self._zl(_ws)
        yt_v = self._yt(_ws)
        a, b = np.cosh(1j * gam * self._n_jct), -zl_v * np.sinh(1j * gam * self._n_jct)
        c, d = -1 / zl_v * np.sinh(1j * gam * self._n_jct), np.cosh(1j * gam * self._n_jct)
        v_in = a + b * yt_v
        i_in = c + (a + d) * yt_v + b * yt_v ** 2
        return v_in / i_in

    def _z_in_cable(self, _ws):
        yg_v = self._yg(_ws)
        zs_v = self._zs(_ws)
        yt_v = self._yt(_ws)

        nj = self._n_jct
        gam = self._gamma(_ws)
        y0 = np.sqrt(yg_v / zs_v)
        a = 1
        # b = 1/(-1j*y0/2*(1 + np.cos(gam*nj))/np.sin(gam*nj) + yg_v*nj/2)
        b = 1 / (nj * yg_v / 2 + 1j * y0 / 2 / np.tan(gam * nj))
        c = 0
        d = 1

        v_in = a + b * yt_v
        i_in = c + (a + d) * yt_v + b * yt_v ** 2
        return v_in / i_in

    def _z_in_perfect(self, _ws):
        yt_v = self._yt(_ws)
        nj = self._n_jct
        a, b, c, d = 1, 1j * nj * self._lj * _ws / (1 - self._lj * self._cj * _ws ** 2), 0, 1
        v_in = a + b * yt_v
        i_in = c + (a + d) * yt_v + b * yt_v ** 2
        return v_in / i_in

    def compute_z_in(self):
        self._ws_init = np.linspace(self._fmin, self._fmax, self._n_point_init) * 2 * np.pi
        self._z_in_init = self._z_in_f(self._ws_init)
        self._sqr_abs_z = np.abs(self._z_in_init) ** 2
        self._log_sqr_abs = np.log(self._sqr_abs_z) / np.log(10)
        self._min_log_sqr_abs_z, self._max_sqr_abs_z = np.nanmin(self._log_sqr_abs), np.nanmax(self._log_sqr_abs)

    def compute_peaks(self):
        peaks_indices, self._peaks_properties = find_peaks(self._sqr_abs_z, height=0, width=1)
        self._peaks_indices = np.array(peaks_indices)
        self._n_mode = self._peaks_indices.shape[0]
        if self._n_mode != 0:
            self._peak_plot = self._ws_init[self._peaks_indices]

    def fit_lorentzian(self, lst_to_fit=None):
        self._list_guess_width = []
        self._wps = []
        self._qs = []
        self._popts = []
        self._pcovs = []

        if self._n_mode == 0:
            return

        reduced = lst_to_fit is not None

        _heights = self._peaks_properties['peak_heights']
        _width_guess = self._peaks_properties["right_ips"] - self._peaks_properties["left_ips"]

        for ind, (p, he, hwg) in enumerate(zip(self._peak_plot, _heights, _width_guess)):
            guess_width = self._ws_init[int(hwg)] + (self._ws_init[1] - self._ws_init[0]) * (hwg % 1)
            if not reduced or (reduced and ind in lst_to_fit):
                tight_ws = np.linspace(p - guess_width * self._guess_width_ratio,
                                       p + guess_width * self._guess_width_ratio,
                                       self._n_point_fit)
                tight_z_in = self._z_in_f(tight_ws)
                tight_abs_z_in = np.abs(tight_z_in) ** 2
                tight_abs_z_in = tight_abs_z_in / np.nanmax(tight_abs_z_in)

                n_tighter = 0

                while (tight_abs_z_in[0] < self._threshold_refinement or tight_abs_z_in[
                    -1] < self._threshold_refinement) \
                        and self._lossless * n_tighter < 1:
                    n_tighter += 1
                    new_peak, new_properties = find_peaks(tight_abs_z_in, height=0, width=1)
                    if len(new_peak) != 1:
                        print(len(new_peak))
                        raise ValueError
                    p = tight_ws[new_peak[0]]
                    hw_guess = new_properties["right_ips"][0] - new_properties["left_ips"][0]
                    guess_width = tight_ws[int(hw_guess)] - tight_ws[0] + (tight_ws[1] - tight_ws[0]) * (hw_guess % 1)

                    if self._show_fit_window:
                        plt.plot(tight_ws / 2 / np.pi, tight_abs_z_in / np.nanmax(tight_abs_z_in))
                        plt.plot([p / 2 / np.pi, p / 2 / np.pi], [0, 1], "r+--")
                        plt.plot([(p - guess_width) / 2 / np.pi, (p + guess_width) / 2 / np.pi], [0.5, 0.5])

                    tight_ws = np.linspace(p - guess_width * self._guess_width_ratio,
                                           p + guess_width * self._guess_width_ratio,
                                           self._n_point_fit)
                    tight_z_in = self._z_in_f(tight_ws)
                    tight_abs_z_in = np.abs(tight_z_in) ** 2
                    tight_abs_z_in = tight_abs_z_in / np.nanmax(tight_abs_z_in)

                    if self._show_fit_window:
                        plt.title(f"Mode {ind + 1}: " + str(p / 2 / np.pi))
                        plt.plot(tight_ws / 2 / np.pi, tight_abs_z_in / np.nanmax(tight_abs_z_in))
                        plt.show()

                self._peak_plot[ind] = p
                self._list_guess_width.append(guess_width)

                if self._lossless:
                    self._wps.append(p)
                    self._qs.append(np.inf)
                else:
                    popt, pcov = curve_fit(lorentzian, tight_ws, tight_abs_z_in, p0=[p, guess_width])
                    self._popts.append(popt)
                    self._pcovs.append(pcov)
                    [wp, width] = popt
                    self._wps.append(wp)
                    self._qs.append(abs(wp / width))
            else:
                self._list_guess_width.append(guess_width)
                self._wps.append(p)
                self._qs.append(np.inf)

    def compute_model_modes(self, lst_to_fit=None):
        self.compute_z_in()
        self.compute_peaks()
        self.fit_lorentzian(lst_to_fit)

    def get_pulsations(self):
        return self._wps

    def get_qs(self):
        return self._qs

    def compute_v_junction(self):
        self._v_points_modes = []
        self._v_diffs_modes = []
        for w_mode in self._wps:
            gam = self._gamma(w_mode)
            v_pnt = np.fromfunction(lambda n: np.sinh(1j * gam * n) - np.sinh(1j * gam * (self._n_jct - n)),
                                    (self._n_jct + 1,), dtype=complex)
            v_pnt /= np.sinh(1j * gam * self._n_jct)
            self._v_points_modes.append(v_pnt[1:-1])
            diffs = v_pnt[:-1] - v_pnt[1:]
            self._v_diffs_modes.append(diffs)

    def compute_phi_zpf(self):
        self._phi_zpf = []
        for v_diff, w_mode in zip(self._v_diffs_modes, self._wps):
            kinetic_energy = np.sum(v_diff ** 2) / self._lj / 2
            alpha = 2 * self._e / self._hbar * np.sqrt(self._hbar * w_mode / 4 / kinetic_energy)
            self._phi_zpf.append(alpha * abs(v_diff))

    def compute_perfect_ground_state_phi_zpf(self):
        self._perfect_ground_state_phi_zpf = 2 * self._e / self._hbar * np.sqrt(self._hbar * self._wps[0]
                                                                                / 2 * self._lj / self._n_jct)

    def get_v_junction(self):
        return self._v_diffs_modes

    def get_phi_zpf(self):
        return self._phi_zpf

    def get_perfect_grd_phi_zpf(self):
        return self._perfect_ground_state_phi_zpf

    def compute_kerr(self):
        self._kerr = np.zeros((self._n_mode, self._n_mode))
        for i in range(self._n_mode):
            for j in range(i, self._n_mode):
                kerr = 0
                for junc in range(self._n_jct):
                    kerr += (self._phi_zpf[i][junc] * self._phi_zpf[j][junc]) ** 2
                kerr *= self._E_J_GHz / 4
                self._kerr[i, j] = kerr
        for i in range(self._n_mode):
            self._kerr[i, i] *= 2

    def get_kerr(self):
        return self._kerr

    def compute_model_quantum(self):
        self.compute_v_junction()
        self.compute_phi_zpf()
        self.compute_perfect_ground_state_phi_zpf()
        self.compute_kerr()

    def compute_model(self, lst_to_fit=None):
        self.compute_model_modes(lst_to_fit)
        self.compute_model_quantum()

    def print_mode_info(self):
        for i, (freq, q) in enumerate(zip(self._wps, self._qs)):
            print(f"Mode : {i + 1} | freq : {freq / 2 / np.pi} | Q : {q:.3e}")

    def plot_z_in(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self._ws_init / 2 / np.pi, self._log_sqr_abs)

        if self._peaks_indices is not None:
            for w in self._ws_init[self._peaks_indices]:
                ax.plot([w / 2 / np.pi, w / 2 / np.pi],
                        [self._min_log_sqr_abs_z, self._max_sqr_abs_z],
                        "+--", color="red")

        ax.set_xlabel("f (GHz)")
        ax.set_ylabel(r"$2 log10(|Z_in|)$")
        ax.grid()
        fig.tight_layout()
        plt.show()

    def plot_z_in_fits(self, mode_number):
        try:
            n = len(mode_number)
        except TypeError:
            n = 0
            print("Mode number should be an iterable.")
        if not all([type(m) == int for m in mode_number]):
            raise TypeError("Mode number should be ints.")

        fig, ax = plt.subplots(1 + n, 1, figsize=(8, 6))
        if len(mode_number) == 0:
            ax = [ax]
        ax[0].plot(self._ws_init / 2 / np.pi, self._log_sqr_abs)

        for w in self._ws_init[self._peaks_indices]:
            ax[0].plot([w / 2 / np.pi, w / 2 / np.pi],
                       [self._min_log_sqr_abs_z, self._max_sqr_abs_z],
                       "+--", color="red")

        ax[0].set_xlabel("f (GHz)")
        ax[0].set_ylabel(r"$2 log10(|Z_in|)$")
        ax[0].grid()
        for i, (mode, gw) in enumerate(zip(mode_number, self._list_guess_width)):
            w_in_fit = np.linspace(self._peak_plot[mode] - gw * 2 / 3,
                                   self._peak_plot[mode] + gw * 2 / 3,
                                   self._n_point_fit)

            sqr_abs_z_in_fit_plt = np.abs(self._z_in_f(w_in_fit)) ** 2
            sqr_abs_z_in_fit_plt = sqr_abs_z_in_fit_plt / np.nanmax(sqr_abs_z_in_fit_plt)
            ax[i + 1].plot(w_in_fit / 2 / np.pi, sqr_abs_z_in_fit_plt, label="Z_in")

            if not self._lossless:
                fit = lorentzian(w_in_fit, *self._popts[mode])
                ax[i + 1].plot(w_in_fit / 2 / np.pi, fit, "--", label="fit")

            ax[i + 1].set_ylabel(r"$|Z_in|^2$")
            ax[i + 1].set_title(f"Mode {mode + 1}")
            ax[i + 1].legend()
            ax[i + 1].grid()
        ax[-1].set_xlabel("f (GHz)")
        fig.tight_layout()
        plt.show()

    def plot_mode_report(self, article=False):
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        if article:
            article_data = np.load("../Article_experimental_data.npy")
            n_plot = len(article_data)
            ax[0].plot(article_data, '+--', label="Article")
        else:
            n_plot = len(self._wps)

        ax[0].plot([w / 2 / np.pi for w in self._wps][:n_plot], "+--", label="Fit")
        ax[1].plot(self._qs[:n_plot], "+--", label="Fit")

        ax[0].set_ylabel("f (GHz)")
        ax[0].legend()
        ax[0].grid()

        ax[1].set_ylabel("Q")
        ax[1].set_xlabel("Mode number")
        ax[1].set_yscale("log")
        ax[1].grid()
        plt.tight_layout()
        plt.show()

    def sweep_params_resistor(self):
        self._variation_params = []
        lens = [len(x) for x in [self._rjp_l, self._rjs_l, self._rg_l, self._rt_l]]
        if sum([lst == 1 for lst in lens]) < 3:
            raise ValueError("Only one parameter should be sweeped.")
        self._sweeped = sum([(lens[i] > 1) * i for i in range(len(lens))])
        for rjp in self._rjp_l:
            for rjs in self._rjs_l:
                for rg in self._rg_l:
                    for rt in self._rt_l:
                        self._variation_params.append([rjp, rjs, rg, rt])

    def sweep_losses_resistor(self, rjp_l, rjs_l, rg_l, rt_l):
        self._rjp_l = rjp_l
        self._rjs_l = rjs_l
        self._rg_l = rg_l
        self._rt_l = rt_l

        self._variations_wps = []
        self._variations_qs = []
        self.sweep_params_resistor()
        to_sweep = tqdm(self._variation_params) if self._show_progression else self._variation_params
        for params in to_sweep:
            [rjp, rjs, rg, rt] = params
            self.init_losses_resistor(rjp, rjs, rg, rt)
            self.compute_z_in()
            self.compute_peaks()
            self.fit_lorentzian()
            self._variations_wps.append(self._wps)
            self._variations_qs.append(self._qs)

    def sweep_params_tand(self):
        self._variation_params = []
        lens = [len(x) for x in [self._tandj_l, self._tandg_l, self._tandt_l]]
        if sum([lst == 1 for lst in lens]) < 2:
            raise ValueError("Only one parameter should be sweeped.")
        self._sweeped = sum([(lens[i] > 1) * i for i in range(len(lens))])
        for tandj in self._tandj_l:
            for tandg in self._tandg_l:
                for tandt in self._tandt_l:
                    self._variation_params.append([tandj, tandg, tandt])

    def sweep_losses_tand(self, tandj_l, tandg_l, tandt_l):
        self._tandj_l = tandj_l
        self._tandg_l = tandg_l
        self._tandt_l = tandt_l

        self._variations_wps = []
        self._variations_qs = []
        self.sweep_params_tand()
        to_sweep = tqdm(self._variation_params) if self._show_progression else self._variation_params
        for params in to_sweep:
            [tandj, tandg, tandt] = params
            self.init_losses_tand(tandj, tandg, tandt)
            self.compute_z_in()
            self.compute_peaks()
            self.fit_lorentzian()
            self._variations_wps.append(self._wps)
            self._variations_qs.append(self._qs)

    def print_sweep_report(self):
        for rjp, rjs, rg, rt, wps, qs in zip(*self._variation_params, self._variations_wps, self._variations_qs):
            print(f"R_jp = {rjp:.3e} | R_js = {rjs:.3e} | R_g = {rg:.3e} | R_t = {rt:.3e}\n")
            for i, (wp, q) in enumerate(zip(wps, qs)):
                print(f"Mode {i + 1} : freq = {wp / 2 / np.pi} | Q = {q:.3e}")
            print("\n--------------------------------------\n")

    def plot_sweep_report(self, mode_number, plot="qs"):
        x_vals = [self._tandj_l, self._tandg_l, self._tandt_l, self._rt_l][self._sweeped]
        try:
            n = len(mode_number)
        except TypeError:
            n = 0
            print("Mode number should be an iterable.")

        fig, axs = plt.subplots(n, 1, figsize=(12, 8))
        if n == 1:
            axs = [axs]

        if plot == "wp":
            var = np.array(self._variations_wps)
            for ax in axs:
                ax.set_ylabel("f (GHz)")
        elif plot == "qs":
            var = np.array(self._variations_qs)
            for ax in axs:
                ax.set_ylabel("Q")
        else:
            raise ValueError(
                "plot variable shoud be either 'wp' for modes' frequency plot or 'qs' for modes' quality factor plot.")

        if not all([type(m) == int for m in mode_number]):
            raise TypeError("Mode number should be ints.")

        for ind_plot, m in enumerate(mode_number):
            axs[ind_plot].plot(x_vals, var[:, m])
            axs[ind_plot].set_yscale("log")
            axs[ind_plot].set_xscale("log")
            axs[ind_plot].set_title(f"Mode {m + 1}")
            axs[ind_plot].grid()

        axs[-1].set_xlabel(["$R_{jp}$", "$R_{js}$", "$R_{g}$", "$R_{t}$"][self._sweeped])
        plt.show()

    def get_variation(self):
        return self._variations_wps, self._variations_qs, self._variation_params

    def data_fit_plot(self, plot_article_data=False, n_plot=9):
        article_data = np.load("../Article_experimental_data.npy")

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.grid()
        self.compute_z_in()
        self.compute_peaks()
        self.fit_lorentzian()
        f_plot = [w / 2 / np.pi for w in self._wps[:n_plot]]
        line, = ax.plot(f_plot, "+--", label="Fit")
        if plot_article_data:
            ax.plot(article_data[:n_plot], "+--", label="Data")
        f_pred = 1 / np.sqrt(self._lj * self._n_jct * self._ct / 2) / 2 / np.pi
        line_pred, = ax.plot([-0.1, 0.1], [f_pred, f_pred], label="Perfect prediction")

        ax.set_title("Mode frequency with respect to mode number")
        ax.set_xlabel('Mode number')
        ax.set_ylabel('Frequency [GHz]')
        ax.legend()
        plt.subplots_adjust(bottom=0.35)

        log_10_vals = [np.log10(self._lj), np.log10(self._cj), np.log10(self._cg), np.log10(self._ct)]
        min_var = [-2] * 4
        max_var = [2] * 4
        pos_slider = [0.25, 0.2, 0.15, 0.1]
        labels = ['log10 Junction inductance [nH]', 'log10 Junction capacity [nF]',
                  'log10 Capacity ground [nF]', 'log10 Resonator capacity [nF]']
        sliders = []

        for init_v, min_v, max_v, pos, lab in zip(log_10_vals, min_var, max_var, pos_slider, labels):
            ax_sli = plt.axes([0.25, pos, 0.65, 0.03])
            sliders.append(Slider(ax=ax_sli,
                                  label=lab,
                                  valmin=init_v + min_v,
                                  valmax=init_v + max_v,
                                  valinit=init_v))

        def update(val):
            vals = [10 ** s.val for s in sliders]
            self.init_params(self._n_jct, *vals)
            self.compute_z_in()
            self.compute_peaks()
            self.fit_lorentzian()
            f_plot = [w / 2 / np.pi for w in self._wps[:n_plot]]
            line.set_ydata(f_plot)
            f_pred = 1 / np.sqrt(vals[0] * self._n_jct * vals[3] / 2) / 2 / np.pi
            line_pred.set_ydata([f_pred, f_pred])
            ymin, ymax = min(f_plot + [f_pred]), max(f_plot + [f_pred])
            ax.set_ylim(ymin - 0.5, ymax + 0.5)
            fig.canvas.draw_idle()

        for s in sliders:
            s.on_changed(update)

        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', hovercolor='0.975')

        def reset(event):
            for _s in sliders:
                _s.reset()

        button.on_clicked(reset)
        plt.show()

    def plot_Q_mode(self, mode, losses_variation, loss_type, resistor_losses=True, progression=True):
        losses = [np.inf, 0, np.inf, np.inf] if resistor_losses else [0] * 3
        qs_plt = []
        for var in tqdm(losses_variation) if progression else losses_variation:
            losses[loss_type] = var
            if resistor_losses:
                self.init_losses_resistor(*losses)
            else:
                self.init_losses_tand(*losses)
            self.compute_model_modes([mode])
            qs_plt.append(self._qs[mode])
        fig, ax = plt.subplots()
        ax.plot(losses_variation, qs_plt)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid()
        plt.show()

    def set_losses_mode(self, Q_target, mode, resistor_losses=True, tolerance=1, progression=True):
        if resistor_losses:
            init = [1e8, 0, 1e8, 1e8]
            lossless = [np.inf, 0, np.inf, np.inf]

            def func(loss, loss_type):
                losses = lossless.copy()
                losses[loss_type] = loss
                self.init_losses_resistor(*losses)
                self.compute_model(lst_to_fit=[mode])
                return Q_target - self._qs[mode]
        else:
            init = [1e-7] * 3
            lossless = [0] * 3

            def func(loss, loss_type):
                losses = lossless.copy()
                losses[loss_type] = loss
                self.init_losses_tand(*losses)
                self.compute_model(lst_to_fit=[mode])
                return abs(Q_target - self._qs[mode])

        res = []
        for ind, init_v in tqdm(enumerate(init)) if progression else enumerate(init):
            m = minimize(func, init_v, args=(ind,), tol=tolerance)
            res.append(m.x[0])
        return res
