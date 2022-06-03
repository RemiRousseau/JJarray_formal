# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 17:01:04 2022

@author: rroussea
"""

import sympy as sp
import numpy as np
from scipy.optimize import minimize


def test_expression(Y_eq, type_test):
    w, s = sp.symbols(r"\omega s", real=True)
    if type_test == "foster":
        Y_eq = Y_eq.subs({w: - sp.I*s})
        var = s
    elif type_test == "resonances":
        Y_eq = Y_eq.subs({s: sp.I*w})
        var = w
    num, deno = sp.fraction(Y_eq)
    deg_num, deg_deno = sp.degree(num, gen=var), sp.degree(deno, gen=var)
    if deg_num != deg_deno+1:
        if deg_deno != deg_num+1:
            print("The degrees of the numerator and denominator do not match the expected values.")
            print("This may not be the wright initial circuit.")
            print(f"Degree numerator : {deg_num} | denominator : {deg_deno}")
        else:
            print("Y_eq might been impedance and not admittance, inversing ...")
            Y_eq = sp.factor(1/Y_eq)
    return Y_eq, num, deno, deg_num, deg_deno


def foster_decompo(Y_eq, big_eq=False, display=False):
    """Calculates Foster decomposition for capacities linked to input & output wires.
    Entry must be an admitance.
    Return L_eq,C_eq,[[L_corr_1,C_corr_1],[L_corr_2,C_corr_2],...]"""
    
    Y_eq, num, deno, deg_num, deg_deno = test_expression(Y_eq, "foster")
    var = sp.symbols(r"s", real = True)
    if display: print(f"Degree numerator : {deg_num}\nDegree denominator : {deg_deno}\n")
    
    if big_eq:
        if display: print("Computing L_eq and C_eq..")
        C_eq, deno_z_ind = sp.div(num, deno)
        C_eq = float(C_eq/var)
        L_eq, num_z_ind = sp.div(deno, deno_z_ind)
        L_eq = float(L_eq/var)
        if display: print(f"L_eq = {L_eq}, C_eq = {C_eq}")
    else:
        L_eq, C_eq = 0, 0
        num_z_ind, deno_z_ind = deno, num

    if num_z_ind == 0:
        return L_eq, C_eq, []
    if display: print("Computing corrections..")
    corr = partial_sum(num_z_ind, deno_z_ind)
    return L_eq, C_eq, corr
    

def partial_sum(num, deno):
    """Returns the partial sum decomposition of the fraction : num/deno
    The degree of num should be one unity more than the one of deno"""

    s = sp.symbols('s', real=True)
    coefs_deno = [float(el) for el in sp.poly(deno).all_coeffs()[::-1]]
    n_coef_deno = len(coefs_deno)
    deno_prime = sum([(i+1)*coefs_deno[i+1]*s**i for i in range(n_coef_deno-1)])
    roots_deno = np.polynomial.polynomial.polyroots(np.array(coefs_deno))
    roots_deno = [roots_deno[2*n] for n in range(n_coef_deno//2)]
    a_l = [abs(float(sp.re(num.subs({s: r})/deno_prime.subs({s: r})))) for r in roots_deno]
    L_l = [2*a/abs(b)**2 for a, b in zip(a_l, roots_deno)]
    C_l = [1/(2*a) for a in a_l]
    corr = [[l, c] for l, c in zip(L_l, C_l)]
    corr.sort(reverse=True)
    return corr


def create_param(x=(15, 7.5e-06, 1e-06, 6.7e-05, 1e9)):
    if len(x) == 4:
        [L_j0_val, C_j0_val, C_g_val, C_0_val], r_val = x, 1e9
    else:
        [L_j0_val, C_j0_val, C_g_val, C_0_val, r_val] = x
    L_j0, C_j0, C_g, C_0, r = sp.symbols("L_j0 C_j0 C_g C_0 R", real=True)
    return {L_j0: L_j0_val, C_j0: C_j0_val, C_g: C_g_val, C_0: C_0_val, r: r_val}


def create_function(z_eq, params):
    w, s, var = sp.symbols(r"\omega s var", real=True)
    subed = sp.factor(z_eq.subs(params))
    subed = subed.subs({w: var, s: var})

    def z_eq_func(_w):
        return complex(subed.subs({var: _w}))

    return z_eq_func


def circuit_resonances(Y_eq, var=sp.symbols(r"\omega", real=True), params=None, full=True, frequency=True, loss=False):
    """Returns the resonant pulsations/frequencies."""
    if params is None:
        L_j0, C_j0, C_g, C_0, r = sp.symbols("L_j0 C_j0 C_g C_0 R", real=True)
        params = {L_j0: 15, C_j0: 7.5e-06, C_g: 1e-06, C_0: 6.7e-05, r: 1e9}
    else:
        C_0 = sp.symbols("C_0", real=True)
    Y_eq, num, deno, deg_num, deg_deno = test_expression(Y_eq, "resonances")
    
    if loss:
        num_tot = sp.expand((num+sp.I*C_0*var*deno).subs(params))
        coefs_num = [complex(el) for el in sp.poly(num_tot).all_coeffs()[::-1]]
    else:
        num_tot = sp.expand((num+sp.I*C_0*var*deno).subs(params)/sp.I)
        coefs_num = [float(el) for el in sp.poly(num_tot).all_coeffs()[::-1]]
        
    roots_num = np.polynomial.polynomial.polyroots(np.array(coefs_num))
    ws = [(np.real(el), np.imag(el)) for el in roots_num]
    ws.sort()
    if not full:
        #ws = [(el[0], el[1]) for el in ws if (el[0] > 0 and el[1] >= 0)]
        #ws.sort()
        ws = duplicate_deleting(ws)
    if frequency:
        return [(el[0]/2/np.pi, el[1]/2/np.pi) for el in ws]
    else:
        return ws


def duplicate_deleting(resonances, threshold=1e-3):
    positives = [(abs(el[0]), abs(el[1])) for el in resonances]
    positives.sort()
    for i in range(len(positives)-1, 0, -1):
        if abs(positives[i][0]-positives[i-1][0]) < threshold:
            positives.pop(i)
    return positives


def resonances_from_foster(L_eq, C_eq, corr, C_0 = 2.6e-3, frequency=True):
    """Returns the resonant pulsations/frequencies reconstructed from the Foster decomposition."""
    var = sp.symbols("s", real=True)
    z_array = sp.factor(L_eq*var + sum([l*var/(1+l*c*(var**2)) for [l, c] in corr]))
    y_tot = sp.factor((C_eq+C_0)*var + 1/z_array)
    num_y_ac_corr, d = sp.fraction(y_tot)
    coefs_num_y = [float(el) for el in sp.poly(num_y_ac_corr).all_coeffs()[::-1]]
    roots_num_y = np.polynomial.polynomial.polyroots(np.array(coefs_num_y))
    w_corr = list(set([round(abs(np.imag(el)), 10) for el in roots_num_y]))
    w_corr.sort()
    if frequency:
        return [el/2/np.pi for el in w_corr]
    return w_corr


def eigenmode_fit(to_fit, y_eq, x0, norm=False):
    n_fit = len(to_fit)
    if norm:
        def cost(x):
            param = create_param(x)
            f_s = circuit_resonances(y_eq, params=param, full=False, frequency=True)
            return sum([(np.sqrt(f_s[i][1]**2+f_s[i][1]**2) - to_fit[i]) ** 2 for i in range(min(len(f_s), n_fit))])
    else:
        def cost(x):
            param = create_param(x)
            f_s = circuit_resonances(y_eq, params=param, full=False, frequency=True)
            return sum([(f_s[i][0] - to_fit[i])**2 for i in range(min(len(f_s), n_fit))])
    res = minimize(cost, x0)
    if not res.success:
        print("Fit failed")
    return res.x, circuit_resonances(y_eq, params=create_param(res.x), full=False, frequency=True)
