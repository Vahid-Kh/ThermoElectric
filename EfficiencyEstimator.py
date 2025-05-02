""" A ThermoElectric Efficiency Estimator scrip:
    1- Calculation based on datasheet estimate of Voltage & Current &  Calculation based on Zeta values
    2- Calculation of Zeta & max Efficiency based on PN junction semiconductor Seebeck effect



    """


"""  1- Calculation based on datasheet estimate of Voltage & Current &  Calculation based on Zeta values   """
import numpy as np
import math

T =  10
CtoK = 273.15
T_C = 5
T_H = 40
T_ave = (T_H+T_C) / 2
T =T_ave
DT = T_H - T_C

""" Data sheet estimates """

V = 0.0473*DT - 0.124
I = 0.0054*DT + 0.1344

W = V*I

print(" | DT : ", DT," | V: ", V," | I : ", I," | W : ", W)


"""  Calculation based on Zeta values """

""" ppppppppp  for p-PbTe """
S_P = .34987391*T + 114.4786318
# R_TEG = 1e-5 * (8e-6*T**2 - 0.007*T + 2.6118)  # Resistivity [Ohm * Meter]
R_P = (-1e-8*T**3 +1e-5*T**2 + 0.0017*T + 0.6186)  # Resistivity [mOhm * cm]
K_P = 8e-6*T**2 - 0.007*T + 2.6118 # Thermal Conductivity [W/m/K]
ZeTa_P = S_P**2 /R_P /K_P * (T+CtoK) /1e7
maxRedEff_P = (np.sqrt(1+ZeTa_P)-1)/(np.sqrt(1+ZeTa_P)+1)
print("T : ", T, " | Seebeck : ", S_P, "| R_TEG : ", R_P, "| K_TEG : ", K_P, "| ZeTa : ", ZeTa_P, "| maxRedEff : ", maxRedEff_P)

""" nnnnnnnnn for n-PbTe """

S_N = -0.293284981*T + -78.40910427

# R_TEG = 1e-5 * (8e-6*T**2 - 0.007*T + 2.6118)  # Resistivity [Ohm * Meter]
R_N = (-2e-9*T**3 +8e-6*T**2 + 0.001*T + 0.2729)  # Resistivity [mOhm * cm]
K_N = 1e-5*T**2 - 0.0096*T + 3.4338 # Thermal Conductivity [W/m/K]
ZeTa_N = S_N**2 /R_N /K_N * (T+CtoK) /1e7
maxRedEff_N = (np.sqrt(1+ZeTa_N)-1)/(np.sqrt(1+ZeTa_N)+1)
print("T : ", T, " | Seebeck : ", S_N, "| R_TEG : ", R_N, "| K_TEG : ", K_N, "| ZeTa : ", ZeTa_N, "| maxRedEff : ", maxRedEff_N)


""" 2- Calculation of Zeta & max Efficiency based on PN junction semiconductor Seebeck effect """
S_N = S_N  # Seebeck of negatively doped
S_P = S_P  # Seebeck of positively doped

ZT_PN = ((S_P-S_N)**2 * (T_ave + CtoK))/((R_N*K_N)**0.5+(R_P*K_P)**0.5)**2 /1e7
# ZT_PN = 1
Eta_Max = (DT/T_H) * ((1+ZT_PN)**2-1)/((1+ZT_PN)**2+T_C/T_H)
Eta_Max_Eff = (np.sqrt(1+ZT_PN)-1)/(np.sqrt(1+ZT_PN)+1)
Eta_Carnot = (1-((273.15+T_C)/(273.15+T_H)))



print("T : ", T, " | ZT_PN : ", ZT_PN, "| Eta_Max : ", Eta_Max, "| EtaMaxEff : ", Eta_Max_Eff, " | Eta_Carnot : ", Eta_Carnot)



