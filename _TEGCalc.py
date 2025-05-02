""" A ThermoElectric Efficiency Estimator scrip:
    1- Calculation based on datasheet estimate of Voltage & Current &  Calculation based on Zeta values
    2- Calculation of Zeta & max Efficiency based on PN junction semiconductor Seebeck effect



    """
import time
import numpy as np
from sympy import *

import math
import matplotlib.pyplot as plt

""" 1A - Efficiency calculation based on datasheet estimate of Voltage & Current  """
T = 10
CtoK = 273.15
T_C = 25
T_H = 100
T_ave = (T_H + T_C) / 2
T = T_ave
DT = T_H - T_C

""" Data sheet estimates """
V = 0.0473 * DT - 0.124
I = 0.0054 * DT + 0.1344

W = V * I

print("From Datasheet   --> | DT : ", DT, " | V: ", V, " | I : ", I, " | W : ", W)

""" 1B - Efficiency calculation based on Zeta values """

""" ppppppppp  for p-PbTe """
S_P = .34987391 * T + 114.4786318
# R_TEG = 1e-5 * (8e-6*T**2 - 0.007*T + 2.6118)  # Resistivity [Ohm * Meter]
R_P = (-1e-8 * T ** 3 + 1e-5 * T ** 2 + 0.0017 * T + 0.6186)  # Resistivity [mOhm * cm]
K_P = 8e-6 * T ** 2 - 0.007 * T + 2.6118  # Thermal Conductivity [W/m/K]
ZeTa_P = S_P ** 2 / R_P / K_P * (T + CtoK) / 1e7
maxRedEff_P = (np.sqrt(1 + ZeTa_P) - 1) / (np.sqrt(1 + ZeTa_P) + 1)
print("For P-PbTe   -->  T : ", T, " | Seebeck : ", S_P, "| R_TEG : ", R_P, "| K_TEG : ", K_P, "| ZeTa : ", ZeTa_P, "| maxRedEff : ",
      maxRedEff_P)

""" nnnnnnnnn for n-PbTe """

S_N = -0.293284981 * T + -78.40910427

# R_TEG = 1e-5 * (8e-6*T**2 - 0.007*T + 2.6118)  # Resistivity [Ohm * Meter]
R_N = (-2e-9 * T ** 3 + 8e-6 * T ** 2 + 0.001 * T + 0.2729)  # Resistivity [mOhm * cm]
K_N = 1e-5 * T ** 2 - 0.0096 * T + 3.4338  # Thermal Conductivity [W/m/K]
ZeTa_N = S_N ** 2 / R_N / K_N * (T + CtoK) / 1e7
maxRedEff_N = (np.sqrt(1 + ZeTa_N) - 1) / (np.sqrt(1 + ZeTa_N) + 1)
print("For N-PbTe we have -->  T : ", T, " | Seebeck : ", S_N, "| R_TEG : ", R_N, "| K_TEG : ", K_N, "| ZeTa : ", ZeTa_N, "| maxRedEff : ",
      maxRedEff_N)

""" 1C - Calculation of Zeta & max Efficiency based on PN junction semiconductor Seebeck effect """
S_N = S_N  # Seebeck of negatively doped
S_P = S_P  # Seebeck of positively doped

ZT_PN = ((S_P - S_N) ** 2 * (T_ave + CtoK)) / ((R_N * K_N) ** 0.5 + (R_P * K_P) ** 0.5) ** 2 / 1e7
# ZT_PN = 1
Eta_Max = (DT / T_H) * ((1 + ZT_PN) ** 2 - 1) / ((1 + ZT_PN) ** 2 + T_C / T_H)
Eta_Max_Eff = (np.sqrt(1 + ZT_PN) - 1) / (np.sqrt(1 + ZT_PN) + 1)
Eta_Carnot = (1 - ((CtoK + T_C) / (CtoK + T_H)))

print("based on PN junction semiconductor --> T : ", T, " | ZT_PN : ", ZT_PN, "| Eta_Max : ", Eta_Max, "| EtaMaxEff : ", Eta_Max_Eff, " | Eta_Carnot : ",
      Eta_Carnot)

""" 2A - THERMAL CONDUCTIVITY SECTION"""
t_inf_1_C = 550       # Top media flow surface tempreture [degC]
t_inf_2_C = 105       # Bottom media flow surface tempreture [degC]
h1 = 40     # Convection from exhaust flow air  [W/m2K]
h2 = 500    # Convection from pressurized water flow [W/m2K]
t_top = 173             # Module Top surface tempreture [degC]
t_bot = 134             # Module Bottom surface tempreture [degC]
n_j = 100               # Number of junctions/pairs within module
l_mod = 2.5e-3      # thickness of module
a_c_s = 1.2e-5          # Surface area of a single PELLET [m2]
d_mod = 0.054        # Module dimensions
k_s = 1.2               # Thermal cond. [W/m.K]
rho_e_s = 1e-4          # 	e,s is the electrical resistivity of the semiconducting material
r_e_eff_mod = 4
r_el_j = rho_e_s * 2 * l_mod / a_c_s  # electrcial resistance TEG
r_el_mod = r_el_j * n_j
s_p_n_eff = 0.1435 # Seebeck coef  V/K
i = 0.37716
t_inf_1	=	823.2  # Top media flow surface tempreture [K]
t_inf_2	=	378.2  # Bottom media flow surface tempreture [K]
r_e_ff = r_el_j*100
rho_jun_e_s = r_e_eff_mod*a_c_s/2/n_j/l_mod
s_sing_junc= s_p_n_eff/n_j/2
num_mod	=	48
r_load	=	400/num_mod


def r_th_eq(l_TEG=l_mod, a_c_s=a_c_s, k_s=k_s, n_j=n_j):  # Equivalent resistance [K/W]
    l = l_TEG   # thickness of TEG
    n = n_j     # Number of junctions within TEG
    k = k_s     # Thermal cond. [W/m.K]
    a = a_c_s   # Surface area of a single PELLET [m2]
    return l / (n * a * k)


def z_teg(s_sing_junc = s_sing_junc, rho_jun_e_s =rho_jun_e_s , k_s=k_s):
    s = s_sing_junc
    rho = rho_jun_e_s
    k = k_s
    return s**2/(rho*k)


err = np.array(np.ones(6)) * np.array(np.random.rand(6)*1e-5)
r_th = r_th_eq(l_mod, a_c_s, k_s, n_j)

"""_____________________________________________________________________________________________________
_________________________________________________________________________________________________________"""
start = time.time()


""" To be found """
# Define the variables
T1, T2, Q1, Q2, I, P = symbols('Q1 Q2 T1 T2 I P')

# Define the equations
eq1 = -Q1 + (T1 - T2) / r_th + (I * s_p_n_eff * T1) - (I ** 2) * r_e_ff
eq2 = -Q2 + (T1 - T2) / r_th + (I * s_p_n_eff * T2) + (I ** 2) * r_e_ff
eq3 = -T1 + -Q1 / (h1 * d_mod ** 2) + t_inf_1
eq4 = -T2 + Q2 / (h2 * d_mod ** 2) + t_inf_2
eq5 = -I  + (P / r_load)**0.5
eq6 = -P  + (I * s_p_n_eff * (T1 - T2) - 2 * I ** 2 * r_e_ff)

# Initial guess for the solution
# NOTE! Intentionally a very far from answer initial guess was used to test how stable finding solution is
initial_guess =  [ 1
                  ,1
                  ,1
                  ,1
                  ,1
                  ,1
                  ]

# Use nsolve to find the solution
solution = nsolve([eq1, eq2, eq3, eq4, eq5, eq6], [T1, T2, Q1, Q2, I, P ], initial_guess)
# print("Solution:", solution)

Q1 = solution[0]-CtoK
Q2 = solution[1]-CtoK
T1 = solution[2]
T2 = solution[3]
I  = solution[4]
P  = solution[5]

print(
    "     T1 = ", Q1,
    "     T2 = ", Q2,
    "     Q1 = ", T1,
    "     Q2 = ", T2,
    "     I  = ", I ,
    "     P  = ", P ,
    )

print(" Initial guess power from each module:  ", P, "  |  Heat difference is : ",Q1-Q2,  "    Zeta: ",z_teg(s_sing_junc, rho_jun_e_s , k_s))

end = time.time()
print("CODE RUNTIME for numerical solver is  : " + str(end - start))
"""_____________________________________________________________________________________________________
_________________________________________________________________________________________________________"""
