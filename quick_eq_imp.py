# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:43:18 2022

@author: rroussea
"""

import sympy as sp
from tqdm import tqdm


def equivalent_impedance(n_jct, mass_capa=True, all_eqs=True, fourier=True, loss=False, impedance=True, messages=True):
    """Computes the equivalent impedance or admitance of an array of Josephson's junctions.
    N_jct : Number of junctions
    mass_capa : Supra islands connected to the mass or to the input/outputs
    all_eqs : If True, returns an array with equivalent impedance for all number of junctions smaller than N_jct
    fourrier : If True, Fourier transform of impedance, else Laplace transform
    loss_par : If True adds parallele loss to capacity
    impedance : if True, return impedance, else return admitance
    messages : toggle prints"""

    if fourier:
        w = sp.symbols(r"\omega", real=True)
        var = sp.I * w
    else:
        s = sp.symbols("s", real=True)
        var = s

    lj, cj, cg, r = sp.symbols("L_j0 C_j0 C_g R", real=True)

    if n_jct < 0:
        print("Error : The number of junction should be an int superior or equal to 1")
        return []
    elif n_jct == 1:
        if loss:
            z_s = 1 / (1 / r + var * cj + 1 / (lj * var))
        else:
            z_s = 1 / (var * cj + 1 / (lj * var))
        if impedance:
            return [sp.factor(z_s)]
        else:
            return [sp.factor(1 / z_s)]

    if mass_capa:
        return mass_capa_z_eq(n_jct, var, all_eqs, loss, impedance, r, cj, lj, cg, messages)
    else:
        return cable_capa_z_eq(n_jct, var, all_eqs, loss, impedance, r, cj, lj, cg, messages)


def extract_data(lines_eq, variable_names, path):
    eq_l = [sp.parse_expr(line) for line in lines_eq]
    dico_var = {}
    file_dict = {}
    for name in variable_names:
        file_dict[name] = open(path + name + ".txt", "r+")
        dico_var[name] = [sp.parse_expr(line) for line in file_dict[name].readlines()]
    return eq_l, dico_var, file_dict


def initialize_file_mass_capa_z_eq():
    path = "data/mass_capa_z_eq_data/"
    files_names = ["D", "Z_eq", "Z_p", "Z_s_d", "Z_s_g"]
    lines = ["2*Z_p_0 + Z_s_0", "", "Z_p_0", "Z_s_0", "Z_s_0"]
    for name, line in zip(files_names, lines):
        file = open(path + name + ".txt", "w")
        file.write(line)
        file.close()


def mass_capa_z_eq(n_jct, var, all_eqs, loss, impedance, r, cj, lj, cg, messages):
    l_j0, c_j0 = sp.symbols("L_j0 C_j0", real=True)
    if n_jct == 2:
        if loss:
            z_s = 1 / (1 / r + var * cj + 1 / (lj * var))
        else:
            z_s = 1 / (var * cj + 1 / (lj * var))
        if impedance:
            return [sp.factor(2 * z_s.subs({lj: l_j0 / 2, cj: c_j0 * 2}))]
        else:
            return [sp.factor(1 / (2 * z_s).subs({lj: l_j0 / 2, cj: c_j0 * 2}))]

    path = "data/mass_capa_z_eq_data/"
    z_s_0, z_p_0 = sp.symbols("Z_s_0 Z_p_0")

    file_z_eq = open(path + "Z_eq.txt", "r+")
    lines_z_eq = file_z_eq.readlines()[1:]

    n_it, n_it_file = (n_jct - 4) // 2, len(lines_z_eq) // 2

    if n_it_file < n_it:
        file_d = open(path + "D.txt", "r+")
        d_l = [sp.parse_expr(line) for line in file_d.readlines()]
        iterable = range(n_it_file, n_it)
        if messages:
            print("Computing additional denominators... ")
            iterable = tqdm(iterable)
        for _ in iterable:
            d_l.append(sp.factor(2 * z_p_0 + z_s_0 - z_p_0 ** 2 / d_l[-1]))
            file_d.write("\n" + str(d_l[-1]))
        file_d.close()

        variable_names = ["Z_s_g", "Z_s_d", "Z_p"]
        z_eq_l, z_dict, file_dict = extract_data(lines_z_eq, variable_names, path)

        if messages:
            print("Computing additional equivalent elements..")
            iterable = tqdm(iterable)
        for n in iterable:
            z_dict["Z_s_g"].append(sp.factor(z_dict["Z_s_g"][-1] + z_dict["Z_s_d"][-1] * z_dict["Z_p"][-1] / d_l[n]))
            z_dict["Z_s_d"].append(sp.factor(z_s_0 + z_dict["Z_s_d"][-1] * z_p_0 / d_l[n]))
            z_dict["Z_p"].append(sp.factor(z_dict["Z_p"][-1] * z_p_0 / d_l[n]))

            for name in variable_names:
                file_dict[name].write("\n" + str(z_dict[name][-1]))

        for name in variable_names:
            file_dict[name].close()

        iterable = range(n_it_file, n_it)
        if messages:
            print("Computing additional equivalent impedance..")
            iterable = tqdm(iterable)
        for n in iterable:
            z_eq = 2 * z_dict["Z_s_g"][n] + sp.factor((2 * z_dict["Z_p"][n] * z_dict["Z_s_d"][n] /
                                                       (z_dict["Z_p"][n] + z_dict["Z_s_d"][n])))
            file_z_eq.write("\n" + str(z_eq))
            z_eq_l.append(z_eq)

            nomi = (2 * z_p_0 * z_dict["Z_p"][n] * (z_p_0 * (z_s_0 + 2 * z_dict["Z_s_d"][n]) +
                                                    z_s_0 * (z_dict["Z_p"][n] + z_dict["Z_s_d"][n])))
            deno = ((z_p_0 + z_dict["Z_p"][n] + z_dict["Z_s_d"][n]) * (z_dict["Z_p"][n] * (4 * z_p_0 + z_s_0) +
                                                                       z_s_0 * (z_p_0 + z_dict["Z_s_d"][n])))
            z_eq = 2 * (z_dict["Z_s_g"][n] + (z_dict["Z_p"][n] * z_dict["Z_s_d"][n] /
                                              (z_p_0 + z_dict["Z_p"][n] + z_dict["Z_s_d"][n]))) + nomi / deno
            file_z_eq.write("\n" + str(z_eq))
            z_eq_l.append(z_eq)

        if all_eqs:
            res = z_eq_l
        else:
            res = [z_eq_l[-1]]

    else:
        if all_eqs:
            res = [sp.parse_expr(line) for line in lines_z_eq[:2 * n_it]]
        else:
            res = [sp.parse_expr(lines_z_eq[n_jct - 4])]

    if not impedance:
        res = [1 / el for el in res]

    file_z_eq.close()

    if loss:
        z_s_val = 1 / (1 / r + var * cj + 1 / (lj * var))
    else:
        z_s_val = 1 / (var * cj + 1 / (lj * var))
    z_p_val = 1 / (var * cg)

    if messages:
        print("Replacing with element's values..")
        iterable = tqdm(res)
    else:
        iterable = res
    final = []
    for i, el in enumerate(iterable):
        final.append(
            sp.factor((el.subs({z_s_0: z_s_val, z_p_0: z_p_val})).subs({lj: l_j0 / (i + 4), cj: c_j0 * (i + 4)})))
    return final


def initialize_file_cable_capa_z_eq():
    path = "data/cable_capa_z_eq_data/"
    files_names = ["D", "Y_eq", "Y_p", "Y_s", "Y_t"]
    lines = ["2*(Y_p_0 + Y_s_0)", "", "Y_p_0", "Y_s_0", "0"]
    for name, line in zip(files_names, lines):
        file = open(path + name + ".txt", "w")
        file.write(line)
        file.close()


def cable_capa_z_eq(n_jct, var, all_eqs, loss, impedance, r, cj, lj, cg, messages):
    l_j0, c_j0 = sp.symbols("L_j0 C_j0", real=True)
    if n_jct == 2:
        if loss:
            y_s = 1 / r + var * cj + 1 / (lj * var)
        else:
            y_s = var * cj + 1 / (lj * var)
        y_p = var * cg
        return [2 * (y_p + y_s).subs({lj: l_j0 / 2, cj: c_j0 * 2})]

    path = "data/cable_capa_z_eq_data/"
    y_s_0, y_p_0 = sp.symbols("Y_s_0 Y_p_0")

    file_y_eq = open(path + "Y_eq.txt", "r+")
    lines_y_eq = file_y_eq.readlines()[1:]

    n_it, n_it_file = n_jct // 2, len(lines_y_eq) // 2

    if n_it_file < n_it:
        file_d = open(path + "D.txt", "r+")
        d_l = [sp.parse_expr(line) for line in file_d.readlines()]
        iterable = range(n_it_file, n_it)
        if messages:
            print("Computing additional denominators... ")
            iterable = tqdm(iterable)
        for _ in iterable:
            d_l.append(sp.factor(2 * (y_p_0 + y_s_0) - y_s_0 ** 2 / d_l[-1]))
            file_d.write("\n" + str(d_l[-1]))
        file_d.close()

        variable_names = ["Y_t", "Y_s", "Y_p"]
        y_eq_l, y_dict, file_dict = extract_data(lines_y_eq, variable_names, path)

        if messages:
            print("Computing additional equivalent elements..")
        for n in iterable:
            y_dict["Y_t"].append(
                sp.factor(y_dict["Y_t"][-1] + 2 * (y_dict["Y_s"][-1] + y_p_0) * y_dict["Y_p"][-1] / d_l[n]))
            y_dict["Y_s"].append(sp.factor((y_dict["Y_s"][-1] + y_p_0) * y_s_0 / d_l[n]))
            y_dict["Y_p"].append(sp.factor(y_p_0 + y_dict["Y_p"][-1] * y_s_0 / d_l[n]))

            for name in variable_names:
                file_dict[name].write("\n" + str(y_dict[name][-1]))

        for name in variable_names:
            file_dict[name].close()

        if messages:
            print("Computing additional equivalent impedance..")
        for n in iterable:
            y_eq_l.append(sp.factor(y_dict["Y_t"][n] + (y_dict["Y_s"][n] + y_dict["Y_p"][n]) / 2))
            file_y_eq.write("\n" + str(y_eq_l[-1]))
            nomi = 2 * y_p_0 * y_dict["Y_p"][n] + y_p_0 * y_s_0 + y_dict["Y_p"][n] * y_s_0 + 2 * y_dict["Y_s"][n] * \
                   y_dict["Y_p"][n] + y_dict["Y_s"][n] * y_s_0
            denomi = y_p_0 + 2 * y_s_0 + y_dict["Y_p"][n] + y_dict["Y_s"][n]
            y_eq_l.append(sp.factor(y_dict["Y_t"][n] + nomi / denomi))
            file_y_eq.write("\n" + str(y_eq_l[-1]))

        if all_eqs:
            res = y_eq_l
        else:
            res = [y_eq_l[-1 - (n_jct % 2 == 0)]]

    else:
        if all_eqs:
            res = [sp.parse_expr(line) for line in lines_y_eq[:n_jct - 1]]
        else:
            res = [sp.parse_expr(lines_y_eq[n_jct - 2])]

    if impedance:
        res = [1 / el for el in res]

    file_y_eq.close()

    if loss:
        y_s_val = 1 / r + var * cj + 1 / (lj * var)
    else:
        y_s_val = var * cj + 1 / (lj * var)

    y_p_val = var * cg

    if messages:
        print("Replacing with element's values..")
        iterable = tqdm(res)
    else:
        iterable = res
    final = []
    for i, el in enumerate(iterable):
        final.append(
            sp.factor((el.subs({y_s_0: y_s_val, y_p_0: y_p_val})).subs({lj: l_j0 / (i + 2), cj: c_j0 * (i + 2)})))
    return final
