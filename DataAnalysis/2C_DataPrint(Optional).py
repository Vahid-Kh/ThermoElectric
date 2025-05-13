import pandas as pd  # Dataframe library

""" ___ CONSTANTS ___ """
step = 1
nMovAve = 1
nVarAve = 30
lowRange = 0
highRange = 1
printInterval = 1000
CtoK = + 273.15

R = "R134a"

""" Choose data from a list of available data"""

""" must be ordered in a where data from Arduiono is 1st and CATMAN 2nd"""
days = [

    "231010 - Energy balance With TEG_ARDUINO",
    "231010 - With TEG_CATMAN",

    "231010 - Energy balance with space(No TEG)_ARDUINO",
    "231010 - With No TEG_CATMAN",

    "231020_With TEG Test2_ARDUINO",
    "231020_Test2_nrows12840_CATMAN",

    "231020_No TEG_ContactHX_ARDUINO",
    "231020_NO TEG_ContactHX_nrows44276_CATMAN",

    "231020_No TEG_SpacedHX_ARDUINO",
    "231020_No TEG_SpacedHX_nrows6418_CATMAN",

    "231020_No TEG_ZeroGap_ARDUINO",
    "231020_ZeroGap_nrows2532_CATMAN",

    "231020_With TEG_ARDUINO",
    "231020_With TEG_nrows8017_CATMAN",

]



def logic(index):
    if index % step == 0:
        return False
    return True


def print_stats(dataframe):
    # Calculate the average, min, and max for each column
    stats_df = dataframe.agg(['min', 'mean', 'max']).transpose()

    # Print the stats in a tabular format
    # print("num\t\tAverage\t\tMin\t\tMax")
    print("=" * 80)
    print("#", '{:>3}'.format("Num"),
          " | ", '{:>25}'.format("Column"),
          " | ", '{:>10}'.format('min'),
          " | ", '{:>10}'.format('mean'),
          " | ", '{:>10}'.format('max'),

          )
    print("_" * 80)
    i_col_counter = 0
    for index, row in stats_df.iterrows():
        # print(i_col_counter ,f"{index}\t\t{row['mean']:.2f}\t\t{row['min']}\t\t{row['max']}")

        print("#", '{:>3}'.format(i_col_counter),
              " | ", '{:>25}'.format(index),
              " | ", '{:>10}'.format(str(round(row['min'], 5))),
              " | ", '{:>10}'.format(str(round(row['mean'], 5))),
              " | ", '{:>10}'.format(str(round(row['max'], 5))),

              )
        i_col_counter += 1


def TEGEfficiencyEstimator(T_C,T_H):
    """ A ThermoElectric Efficiency Estimator scrip:
        1- Calculation based on datasheet estimate of Voltage & Current &  Calculation based on Zeta values
        2- Calculation of Zeta & max Efficiency based on PN junction semiconductor Seebeck effect
        """

    """  1- Calculation based on datasheet estimate of Voltage & Current &  Calculation based on Zeta values   """
    import numpy as np

    CtoK = 273.15
    T_C = T_C
    T_H = T_H

    T_ave = (T_H + T_C) / 2
    T=T_ave
    DT = T_H - T_C

    """ Data sheet estimates """

    V = 0.0473 * DT - 0.124
    I = 0.0054 * DT + 0.1344

    W = V * I

    """  Calculation based on Zeta values """

    """ ppppppppp  for p-PbTe """
    S_P = .34987391 * T + 114.4786318
    # R_TEG = 1e-5 * (8e-6*T**2 - 0.007*T + 2.6118)  # Resistivity [Ohm * Meter]
    R_P = (-1e-8 * T ** 3 + 1e-5 * T ** 2 + 0.0017 * T + 0.6186)  # Resistivity [mOhm * cm]
    K_P = 8e-6 * T ** 2 - 0.007 * T + 2.6118  # Thermal Conductivity [W/m/K]
    ZeTa_P = S_P ** 2 / R_P / K_P * (T + CtoK) / 1e7
    maxRedEff_P = (np.sqrt(1 + ZeTa_P) - 1) / (np.sqrt(1 + ZeTa_P) + 1)
    # print("T : ", T, " | Seebeck : ", S_P, "| R_TEG : ", R_P, "| K_TEG : ", K_P, "| ZeTa : ", ZeTa_P, "| maxRedEff : ",maxRedEff_P)

    """ nnnnnnnnn for n-PbTe """

    S_N = -0.293284981 * T + -78.40910427

    # R_TEG = 1e-5 * (8e-6*T**2 - 0.007*T + 2.6118)  # Resistivity [Ohm * Meter]
    R_N = (-2e-9 * T ** 3 + 8e-6 * T ** 2 + 0.001 * T + 0.2729)  # Resistivity [mOhm * cm]
    K_N = 1e-5 * T ** 2 - 0.0096 * T + 3.4338  # Thermal Conductivity [W/m/K]
    ZeTa_N = S_N ** 2 / R_N / K_N * (T + CtoK) / 1e7
    maxRedEff_N = (np.sqrt(1 + ZeTa_N) - 1) / (np.sqrt(1 + ZeTa_N) + 1)
    # print("T : ", T, " | Seebeck : ", S_N, "| R_TEG : ", R_N, "| K_TEG : ", K_N, "| ZeTa : ", ZeTa_N, "| maxRedEff : ", maxRedEff_N)

    """ 2- Calculation of Zeta & max Efficiency based on PN junction semiconductor Seebeck effect """
    S_N = S_N  # Seebeck of negatively doped
    S_P = S_P  # Seebeck of positively doped

    ZT_PN = ((S_P - S_N) ** 2 * (T_ave + CtoK)) / ((R_N * K_N) ** 0.5 + (R_P * K_P) ** 0.5) ** 2 / 1e7
    # ZT_PN = 1
    Eta_Max = (DT / T_H) * ((1 + ZT_PN) ** 2 - 1) / ((1 + ZT_PN) ** 2 + T_C / T_H)
    Eta_Max_Eff = (np.sqrt(1 + ZT_PN) - 1) / (np.sqrt(1 + ZT_PN) + 1)
    Eta_Carnot = (1 - ((273.15 + T_C) / (273.15 + T_H)))

    print("THEORITICAL - DT : ", '{:>5}'.format(round(DT,3)),
          " | V :  ", '{:>5}'.format(round(V, 3)),
          " | I :    ", '{:>5}'.format(round(I, 3)),
          " | W :    ", '{:>5}'.format(round(W, 3)),
          )

    print("THEORITICAL - T : ", '{:>5}'.format(round(T,3)),
          " | Seebeck : ", '{:>5}'.format(round((S_N+S_P)/2, 3)),
          " | R_TEG  : ", '{:>5}'.format(round((R_N+R_P)/2, 3)),
          " | K_TEG  : ", '{:>5}'.format(round((K_N+K_P)/2, 3)),
          " | ZeTa  :  ", '{:>5}'.format(round((ZeTa_N+ZeTa_P)/2, 3)),
          " | maxRedEff  : ", '{:>5}'.format(round((maxRedEff_N+maxRedEff_P)/2, 3)),
          )

    print("THEORITICAL - T : ", '{:>6}'.format(round(T,3)),
          " | ZT_PN : ", '{:>5}'.format(round(ZT_PN, 3)),
          " | Eta_Max : ", '{:>5}'.format(round(Eta_Max, 3)),
          " | EtaMaxEff : ", '{:>5}'.format(round(Eta_Max_Eff, 3)),
          " | Eta_Carnot : ", '{:>5}'.format(round(Eta_Carnot, 3)),
          )
#DROP THE COLUMNS NOT USED

for datanum in range(len(days)):
    print(" ____________________- ", days[datanum], " _____________________-              ")
    # print(" ")


    dfRaw = pd.read_csv('Data/Clean/' + days[datanum] + ".txt", skiprows=lambda x: logic(x))
    """ To take a portion of dataframe """

    df = dfRaw[int(len(dfRaw) * lowRange):int(len(dfRaw) * highRange)]

    index = []
    """ PLOT """
    print(df.columns)
    try:
        df.drop(["MX1601_CH_11_[V]", "MX1601_CH_12_[V]", "MX1601_CH_13_[mA]", "MX1601_CH_14_[mA]",], axis=1, inplace=True )

    except:
        print("No column dropped")
    print(df.columns)
    print_stats(df)




