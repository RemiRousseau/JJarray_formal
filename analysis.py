# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 17:01:04 2022

@author: rroussea
"""

import sympy as sp
import numpy as np

def test_expression(Y_eq,type_test):
    w,s = sp.symbols(r"\omega s", real = True)
    if type_test == "foster":
        Y_eq = Y_eq.subs({w: - sp.I*s})
    elif type_test == "resonances":
        Y_eq = Y_eq.subs({w:s})
    num,deno = sp.fraction(Y_eq)
    deg_num, deg_deno = sp.degree(num,gen = s), sp.degree(deno,gen = s)
    if deg_num != deg_deno+1 :
        if deg_deno != deg_num+1 :
            print("The degrees of the numerator and denominator do not match the expected values, this may not be the wright initial circuit.")
            print(f"Degree numerator : {deg_num} | denominator : {deg_deno}")
        else:
            print("Y_eq might been impedance and not admittance, inversing ...")
            Y_eq = 1/Y_eq
    return Y_eq,num,deno,deg_num,deg_deno


def foster_decompo_1(Y_eq, big_eq = False, display = False):
    """Calculates Foster decomposition for capacities linked to input & output wires.
    Entry must be an admitance.
    Return L_eq,C_eq,[[L_corr_1,C_corr_1],[L_corr_2,C_corr_2],...]"""
    
    Y_eq,num,deno,deg_num,deg_deno = test_expression(Y_eq,"foster")
    var = sp.symbols(r"s", real = True)
    if display : print(f"Degree numerator : {deg_num}\nDegree denominator : {deg_deno}\n")
    
    if big_eq :
        if display : print("Computing L_eq and C_eq..")
        C_eq, deno_z_ind = sp.div(num,deno)
        C_eq = float(C_eq/var)
        L_eq, num_z_ind = sp.div(deno,deno_z_ind)
        L_eq = float(L_eq/var)
        if display : print(f"L_eq = {L_eq}, C_eq = {C_eq}")
    else:
        L_eq,C_eq = 0,0
        num_z_ind,deno_z_ind = deno,num

    if num_z_ind == 0:
        return L_eq,C_eq,[]
    if display : print("Computing corrections..")
    corr = partial_sum(num_z_ind,deno_z_ind)
    return L_eq,C_eq,corr

def foster_decompo_2(Y_eq):
    Y_eq,num,deno,deg_num,deg_deno = test_expression(Y_eq,"foster")
    print(f"deg_num : {deg_num} | deg_deno : {deg_deno}")
    var = sp.symbols(r"s", real = True)
    #C_eq,num_n = sp.div(num,deno)
    #C_eq = float(C_eq/var)
    #C_eq = 0
    print(f"deg_num : {sp.degree(num,gen = var)} | deg_deno : {sp.degree(deno,gen = var)}")
    corr = partial_sum(var*num,var*deno)
    return 0,0,[[el[1],el[0]] for el in corr]
    

def partial_sum(num,deno):
    """Returns the partial sum decomposition of the fraction : num/deno
    The degree of num should be one unity more than the one of deno"""

    s = sp.symbols('s', real = True)
    coefs_deno = [float(el) for el in sp.poly(deno).all_coeffs()[::-1]]
    n_coef_deno = len(coefs_deno)
    deno_prime = sum([(i+1)*coefs_deno[i+1]*s**i for i in range(n_coef_deno-1)])
    roots_deno = np.polynomial.polynomial.polyroots(np.array(coefs_deno))
    roots_deno = [roots_deno[2*n] for n in range(n_coef_deno//2)]
    a_l = [abs(float(sp.re(num.subs({s:r})/deno_prime.subs({s:r})))) for r in roots_deno]
    L_l = [2*a/abs(b)**2 for a,b in zip(a_l,roots_deno)]
    C_l = [1/(2*a) for a in a_l]
    corr = [[l,c] for l,c in zip(L_l,C_l)]
    corr.sort(reverse = True)
    return corr

def circuit_resonances(Y_eq, C_0 = 2.6e-3,full = True):
    Y_eq,num,deno,deg_num,deg_deno = test_expression(Y_eq,"resonances")
    var = sp.symbols(r"s", real = True)
    if C_0 == 0 : num_tot = num
    else : num_tot = sp.expand(num+deno*C_0*var)
    if sp.conjugate(num_tot) == -num_tot:
        num_tot = sp.im(num_tot)
    coefs_num = [complex(el) for el in sp.poly(num_tot).all_coeffs()[::-1]]
    roots_num = np.polynomial.polynomial.polyroots(np.array(coefs_num))
    freqs = [(np.real(el),np.imag(el)) for el in roots_num]
    freqs.sort()
    if full:
        return freqs
    else:
        return [el for el in freqs if (el[0]>0 and el[1]>0)]
    
def circuit_resonances_sympy(Y_eq, C_0 = 2.6e-3,full = True):
    Y_eq,num,deno,deg_num,deg_deno = test_expression(Y_eq,"resonances")
    var = sp.symbols(r"s", real = True)
    if C_0 == 0 : num_tot = num
    else : num_tot = sp.expand(num+deno*C_0*var)
    if sp.conjugate(num_tot) == -num_tot:
        num_tot = sp.im(num_tot)
    roots_num = sp.nroots(sp.poly(num_tot),n=200)
    freqs = [(float(sp.re(el)),float(sp.im(el))) for el in roots_num]
    freqs.sort()
    if full:
        return freqs
    else:
        return [el for el in freqs if (el[0]>0 and el[1]>0)]

def resonances_from_foster_1(L_eq,C_eq,corr, C_0 = 2.6e-3) :
    var = sp.symbols("s",real = True)
    z_array = sp.factor(L_eq*var + sum([l*var/(1+l*c*(var**2)) for [l,c] in corr]))
    y_tot = sp.factor((C_eq+C_0)*var + 1/z_array)
    num_y_ac_corr,d = sp.fraction(y_tot)
    coefs_num_y = [float(el) for el in sp.poly(num_y_ac_corr).all_coeffs()[::-1]]
    roots_num_y = np.polynomial.polynomial.polyroots(np.array(coefs_num_y))
    freqs_corr = list(set([round(abs(np.imag(el)),10) for el in roots_num_y]))
    freqs_corr.sort()
    return freqs_corr

def resonances_from_foster_2(L_eq,C_eq,corr, C_0 = 2.6e-3) :
    var = sp.symbols("s",real = True)
    y_tot = sp.factor((C_eq+C_0)*var + sum([c*var/(1+l*c*var**2) for [l,c] in corr]))
    print(y_tot)
    num_y_ac_corr,d = sp.fraction(y_tot)
    coefs_num_y = [float(el) for el in sp.poly(num_y_ac_corr).all_coeffs()[::-1]]
    roots_num_y = np.polynomial.polynomial.polyroots(np.array(coefs_num_y))
    freqs_corr = list(set([round(abs(np.imag(el)),10) for el in roots_num_y]))
    freqs_corr.sort()
    return freqs_corr
