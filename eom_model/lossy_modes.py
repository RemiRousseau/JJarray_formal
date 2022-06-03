import numpy as np
from scipy.optimize import newton


class Array:
    def __init__(self, n_jct, params):
        """
        n_jct : number of junctions in the array
        param_array : [Lj, Cj, Rjs, Rjp, Cg, Rg, Ct] each value corresponds to individual junction parameters
        Lj : junction impedance
        Cj : junction capacity (to be computed with the plasma frequency wp=(LjCj)**-1
        Rjs : junction resistance (in series with Cj) | put to 0 to ignore
        Rjp : junction resistance (in parallel with junction) |
        Cg : capacity to ground of the inter-junction supraconductor island
        Rg : resistance to ground (in parallel with Cg)

        Ct : resonator capacity, can be set to None for the bare array frequencies

        Resistance can be set to 0 (for Rjs) or np.inf (for Rg and Rjp) to ignore.


             |--Cj--Rjs--|         |--Cj--Rjs--|
             |           |         |           |
        ...--|----Lj-----|---------|----Lj-----|-- ...
             |           |    |    |           |
             |----Rjp----|  -----  |----Rjp----|
                            |   |
                            Cg  Rg
                            |   |
                            -----
                              |
        """
        [self._lj, self._cj, self._rjs, self._rjp, self._cg, self._rg, self._ct] = params
        self._n_mode = (n_jct - 1) // 2
        self._mods = [1 - np.cos(np.pi * k / n_jct) for k in range(1, self._n_mode + 1)]
        self._ck = [self._cg / 2 + self._cj * m for m in self._mods]
        self._lk = [self._lj / m for m in self._mods]
        self._rk = []
        self._modes_0 = [1 / np.sqrt(lk * c) for lk, c in zip(self._lk, self._ck)]
        self._zk = [np.sqrt(lk / ck) / 2 for lk, ck in zip(self._lk, self._ck)]

        self._modes_res = []

        if self._rjs == 0 and self._rjp == np.inf and self._rg == np.inf:
            self._q0 = []
        else:
            self._q0 = self.compute_q(self._modes_0)
        self._q_res = []

    def get_bare_modes(self, frequency=True):
        if frequency:
            return [m / 2 / np.pi for m in self._modes_0]
        return self._modes_0

    def get_bare_q(self):
        return self._q0

    def get_resonator_modes(self, frequency=True):
        if frequency:
            return [m / 2 / np.pi for m in self._modes_res]
        return self._modes_res

    def get_resonator_q(self):
        return self._q_res

    def get_bare_modes_rlc(self):
        self._rk = [1/2/self._rg + m*(1/self._rjp + self._cj**2*self._rjs*wk**2)
                    for m, wk in zip(self._mods, self._modes_0)]
        return self._rk, self._lk, self._ck

    def compute_q(self, freqs):
        return [(m / self._lj + wk ** 2 * (self._cg / 2 + self._cj * m)) / (2 * wk * (
                1 / (2 * self._rg) + m * (1 / self._rjp + self._cj ** 2 * self._rjs * wk ** 2))) for m, wk in
                zip(self._mods, freqs)]

    def resonator_correction(self, ct=None, n_corr=None):
        if n_corr is None:
            n_corr = self._n_mode
        if ct is not None:
            self._ct = ct
        self._compute_resonator_correction(n_corr)

    def _compute_resonator_correction(self, n_corr):
        # Even mode computation
        func = lambda wkc, _k, _wk0, _zk: np.tan(wkc * _k * np.pi / _wk0 / 2) + wkc * self._ct * _zk
        func_p = lambda wkc, _k, _wk0, _zk: _k * np.pi / _wk0 / 2 * (
                1 + (np.tan(wkc * _k * np.pi / _wk0 / 2)) ** 2) + self._ct * _zk
        even_modes = []
        for k in range(1, n_corr // 2 + 1):
            wk0 = self._modes_0[2 * k - 1]
            zk = self._zk[2 * k - 1]
            x0 = wk0 / (2 * k) + 1
            try:
                root, root_af = 0, newton(func, x0, fprime=func_p, args=(2 * k, wk0, zk,))
                while root_af < wk0:
                    root = root_af
                    x0 += 2 * wk0 / (2 * k)
                    root_af = newton(func, x0, fprime=func_p, args=(2 * k, wk0, zk,))
            except:
                print(f"Newton method failed to converge. Mode {2 * k} set to 0.")
                root = 0
            even_modes.append(root)

        # Odd modes computation
        func = lambda wkc, _k, _wk0, _zk: np.tan(wkc * _k * np.pi / _wk0 / 2) - 1 / (wkc * self._ct * _zk)
        func_p = lambda wkc, _k, _wk0, _zk: _k * np.pi / _wk0 / 2 * (
                1 + (np.tan(wkc * _k * np.pi / _wk0 / 2)) ** 2) + 1 / (self._ct * _zk * wkc ** 2)
        odd_modes = []
        for k in range(0, (n_corr + 1) // 2):
            wk0 = self._modes_0[2 * k]
            zk = self._zk[2 * k]
            try:
                root = newton(func, wk0 - 1 / (2 * k + 1), fprime=func_p, args=(2 * k + 1, wk0, zk,))
            except:
                print(f"Newton method failed to converge. Mode {2 * k + 1} set to 0.")
                root = 0
            odd_modes.append(root)

        self._modes_res = []
        for odd, even in zip(odd_modes, even_modes):
            self._modes_res.append(odd)
            self._modes_res.append(even)
        if n_corr % 2 == 1:
            self._modes_res.append(odd_modes[-1])
        self._q_res = self.compute_q(self._modes_res)
        print("Resonators Q are under evaluated, calculation to modify")

    def isolated_resonance(self):
        """Test if modes are separated enough to be distinguishable."""
        width = [_w / _q for _w, _q in zip(self._modes_res, self._q_res)]
        _overlap = [x1 - x0 > w0 + w1 for x0, x1, w0, w1 in zip(self._modes_res, self._modes_res[1:], width, width[1:])]
        if all(_overlap):
            return self._n_mode
        else:
            return _overlap.index(False)


