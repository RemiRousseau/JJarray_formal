# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:43:18 2022

@author: rroussea
"""

import sympy as sp
from tqdm import tqdm 

def equivalent_impedance (N_jct, mass_capa = True, all_eqs = True, fourier = True,loss = False, impedance = True, l_j_0 = 15, nu_0 = 15, c_g = 10**-2,r = 10**9,messages = True):
    """Computes the equivalent impedance or admitance of an array of Josephson's junctions.
    N_jct : Number of junctions
    mass_capa : Supra islands connected to the mass or to the input/outputs
    all_eqs : If True, returns an array with equivalent impedance for all number of junctions smaller than N_jct (only odd if mass_capa = False)
    fourrier : If True, Fourier transform of impedance, else Laplace transform
    loss_par : If True adds parallele loss to capacity
    impedance : if True, return impedance, else return admitance
    l_j_0 : wanted equivalent inductance (in nH)
    nu_0 : plasma frequency (in GHz)
    c_g : parasitic capacity (in pF)
    r : resistance (in Ohm) 
    messages : toggle prints"""
    
    if fourier:
        w = sp.symbols(r"\omega",real = True)
        var = sp.I*w
    else:
        s = sp.symbols("s",real = True)
        var = s
    
    if all_eqs :
        l_j = sp.symbols("L_J")
    else:
        l_j = l_j_0/N_jct
    
    c_j = 1/(l_j*nu_0**2)
    
    if N_jct<0:
        print("Error : The number of junction should be an int superior or equal to 1")
        return []
    elif N_jct == 1:
        if loss :
            Z_s = 1/(1/r + var*c_j + 1/(l_j*var))
        else :
            Z_s = 1/(var*c_j + 1/(l_j*var))
        if impedance :
            return [Z_s]
        else: 
            return [1/Z_s]
    
    if mass_capa:
        return mass_capa_z_eq(N_jct, var, all_eqs, loss, impedance, r, c_j, l_j, l_j_0, c_g, messages)
    else:
        return cable_capa_z_eq(N_jct, var, all_eqs, loss, impedance, r, c_j, l_j, l_j_0, c_g, messages)

def initialize_file_mass_capa_z_eq():
    path = "data/mass_capa_z_eq_data/"
    files_names = ["D","Z_eq","Z_p","Z_s_d","Z_s_g"]
    lines = ["2*Z_p_0 + Z_s_0","","Z_p_0","Z_s_0","Z_s_0"]
    for name,line in zip(files_names,lines):
        file = open(path+name+".txt","w")
        file.write(line)
        file.close()

def clean_up_mass_capa_z_eq():
    path = "data/mass_capa_z_eq_data/"
    files_names = ["D","Z_eq","Z_p","Z_s_d","Z_s_g"]
    for name in files_names:
        file = open(path+name+".txt","r")
        print(f"{name} : {str(len(file.readlines()))}\n")
        file.close()
        
    
def mass_capa_z_eq(N_jct, var, all_eqs, loss, impedance, r, c_j, l_j, l_j_0, c_g, messages):
    path = "data/mass_capa_z_eq_data/"
    Z_s_0, Z_p_0 = sp.symbols("Z_s_0 Z_p_0")
    
    file_z_eq = open(path + "Z_eq.txt","r+")
    lines_z_eq = file_z_eq.readlines()[1:]
    
    if len (lines_z_eq) < N_jct-2:
        file_D = open(path + "D.txt","r+")
        D_l = [sp.parse_expr(line) for line in file_D.readlines()]
        iterable = range(N_jct-3-len(D_l))
        if messages :
            print("Computing additional denominators... ")
            iterable = tqdm(iterable)
        for _ in iterable :
            D_l.append(sp.factor(2*Z_p_0+Z_s_0-Z_p_0**2/D_l[-1]))
            file_D.write("\n"+str(D_l[-1]))
        file_D.close()
        
        variable_names = ["Z_s_g","Z_s_d","Z_p"]
    
        z_eq_l = [sp.parse_expr(line) for line in lines_z_eq]
        Z_dict = {}
        file_dict= {}
        for name in variable_names:
            file_dict[name] = open(path+name+".txt","r+")
            Z_dict[name] = [sp.parse_expr(line) for line in file_dict[name].readlines()]
        
        iterable = range(len(Z_dict["Z_s_g"])-1,N_jct - 3)
        if messages : 
            print("Computing additional equivalent elements..")
            iterable = tqdm(iterable)
        for n in iterable:
            Z_dict["Z_s_g"].append(sp.factor(Z_dict["Z_s_g"][-1] + Z_dict["Z_s_d"][-1]*Z_dict["Z_p"][-1]/D_l[n]))
            Z_dict["Z_s_d"].append(sp.factor(Z_s_0 + Z_dict["Z_s_d"][-1]*Z_p_0/D_l[n]))
            Z_dict["Z_p"].append(sp.factor(Z_dict["Z_p"][-1]*Z_p_0/D_l[n]))
            
            for name in variable_names:
                file_dict[name].write("\n"+str(Z_dict[name][-1]))
        
        for name in variable_names:
            file_dict[name].close()
        
        iterable = range(len(z_eq_l)+3,N_jct+1)
        if messages : 
            print("Computing additional equivalent impedance..")
            iterable = tqdm(iterable)
        for n in iterable:
            z_eq = (Z_dict["Z_s_g"][n-3] + Z_s_0 + Z_dict["Z_s_d"][n-3]*(Z_dict["Z_p"][n-3]+Z_p_0)/(Z_dict["Z_s_d"][n-3]+Z_dict["Z_p"][n-3]+Z_p_0))
            z_eq = sp.factor(z_eq)
            file_z_eq.write("\n"+str(z_eq))
            z_eq_l.append(z_eq)
        
        if all_eqs:
            res = z_eq_l
        else:
            res = [z_eq_l[-1]]
    
    else:
        if all_eqs:
            res =[sp.parse_expr(line) for line in lines_z_eq[:N_jct-2]]
        else:
            res =[sp.parse_expr(lines_z_eq[N_jct-3])]
            
    if not impedance :
        res =[1/el for el in res]
    
    file_z_eq.close()
    
    if loss :
        Z_s_val = 1/(1/r + var*c_j + 1/(l_j*var))
    else :
        Z_s_val = 1/(var*c_j + 1/(l_j*var))
    Z_p_val = 1/(var*c_g)
    
    if messages :
        print("Replacing with element's values..")
        iterable = tqdm(res)
    else:
        iterable = res
    final = []
    for i,el in enumerate(iterable) :
        final.append(sp.factor((el.subs({Z_s_0:Z_s_val,Z_p_0:Z_p_val})).subs({l_j : l_j_0/(i+3)})))
    return final

def initialize_file_cable_capa_z_eq():
    path = "data/cable_capa_z_eq_data/"
    files_names = ["D","Y_eq","Y_p","Y_s","Y_t"]
    lines = ["2*(Y_p_0 + Y_s_0)","","Y_p_0","Y_s_0","0"]
    for name,line in zip(files_names,lines):
        file = open(path+name+".txt","w")
        file.write(line)
        file.close()

def clean_up_cable_capa_z_eq():
    path = "data/cable_capa_z_eq_data/"
    files_names = ["D","Y_eq","Y_p","Y_s","Y_t"]
    for name in files_names:
        file = open(path+name+".txt","r")
        print(f"{name} : {str(len(file.readlines()))}\n")
        file.close()

def cable_capa_z_eq(N_jct, var, all_eqs, loss, impedance, r, c_j, l_j, l_j_0, c_g, messages):
    path = "data/cable_capa_z_eq_data/"
    Y_s_0, Y_p_0 = sp.symbols("Y_s_0 Y_p_0")
    
    file_y_eq = open(path + "Y_eq.txt","r+")
    lines_y_eq = file_y_eq.readlines()[1:]
    
    N_it,N_it_file = N_jct//2,len(lines_y_eq)//2
    
    if N_it_file < N_it :
        file_D = open(path + "D.txt","r+")
        D_l = [sp.parse_expr(line) for line in file_D.readlines()]
        iterable = range(N_it-N_it_file)
        if messages :
            print("Computing additional denominators... ")
            iterable = tqdm(iterable)
        for _ in iterable :
            D_l.append(sp.factor(2*(Y_p_0+Y_s_0)-Y_s_0**2/D_l[-1]))
            file_D.write("\n"+str(D_l[-1]))
        file_D.close()
        
        variable_names = ["Y_t","Y_s","Y_p"]
    
        y_eq_l = [sp.parse_expr(line) for line in lines_y_eq]
        Y_dict = {}
        file_dict= {}
        for name in variable_names:
            file_dict[name] = open(path+name+".txt","r+")
            Y_dict[name] = [sp.parse_expr(line) for line in file_dict[name].readlines()]
        
        iterable = range(N_it_file,N_it)
        if messages : 
            print("Computing additional equivalent elements..")
            iterable = tqdm(iterable)
        for n in iterable:
            Y_dict["Y_t"].append(sp.factor(Y_dict["Y_t"][-1] + 2*(Y_dict["Y_s"][-1]+Y_p_0)*Y_dict["Y_p"][-1]/D_l[n]))
            Y_dict["Y_s"].append(sp.factor((Y_dict["Y_s"][-1] + Y_p_0)*Y_s_0/D_l[n]))
            Y_dict["Y_p"].append(sp.factor(Y_p_0 + Y_dict["Y_p"][-1]*Y_s_0/D_l[n]))
            
            for name in variable_names:
                file_dict[name].write("\n"+str(Y_dict[name][-1]))
        
        for name in variable_names:
            file_dict[name].close()
        
        iterable = range(N_it_file,N_it)
        if messages : 
            print("Computing additional equivalent impedance..")
            iterable = tqdm(iterable)
        for n in iterable:
            y_eq_l.append(sp.factor(Y_dict["Y_t"][n]+(Y_dict["Y_s"][n] + Y_dict["Y_p"][n])/2))
            file_y_eq.write("\n"+str(y_eq_l[-1]))
            nomi = 2*Y_p_0*Y_dict["Y_p"][n] + Y_p_0*Y_s_0 + Y_dict["Y_p"][n]*Y_s_0 + 2*Y_dict["Y_s"][n]*Y_dict["Y_p"][n] + Y_dict["Y_s"][n]*Y_s_0
            denomi = Y_p_0 + 2* Y_s_0 + Y_dict["Y_p"][n] + Y_dict["Y_s"][n]
            y_eq_l.append(sp.factor(Y_dict["Y_t"][n] +  nomi/denomi))
            file_y_eq.write("\n"+str(y_eq_l[-1]))
        
        if all_eqs:
            res = y_eq_l
        else:
            res = [y_eq_l[-1-(N_jct%2==0)]]
    
    else:
        if all_eqs:
            res =[sp.parse_expr(line) for line in lines_y_eq[:N_jct-1]]
        else:
            res =[sp.parse_expr(lines_y_eq[N_jct-2])]
            
    if impedance :
        res =[1/el for el in res]
    
    file_y_eq.close()
    
    if loss:
        Y_s_val = 1/r + var*c_j + 1/(l_j*var)
    else:
        Y_s_val = var*c_j + 1/(l_j*var)
    
    Y_p_val = var*c_g
    
    if messages :
        print("Replacing with element's values..")
        iterable = tqdm(res)
    else:
        iterable = res
    final = []
    for i,el in enumerate(iterable) :
        final.append(sp.factor((el.subs({Y_s_0:Y_s_val,Y_p_0:Y_p_val})).subs({l_j : l_j_0/(i+2)})))
    return final