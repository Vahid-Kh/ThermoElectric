from CoolProp.CoolProp import PropsSI
import pandas as pd  # Dataframe library
import numpy as np  # Scientific computing with nD object support
import random as rd
import math
from functions import *
from scipy.signal import butter, lfilter, freqz

from scipy.signal import correlate


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
    import math

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

    print("THEORETICAL :  ", '{:>35}'.format(" DT = " + str(round(DT, 3))),
          " | V :    ", '{:>9}'.format(round(V, 3)),
          " | I :     ", '{:>9}'.format(round(I, 3)),
          " | W :    ", '{:>9}'.format(round(W, 3)),
          )

    print("THEORETICAL :  ", '{:>35}'.format(" T = " + str(round(T,3))),
          " | Seebeck = ", '{:>5}'.format(round((S_N+S_P)/2, 3)),
          " | R_TEG = ", '{:>9}'.format(round((R_N+R_P)/2, 3)),
          " | K_TEG = ", '{:>6}'.format(round((K_N+K_P)/2, 3)),
          " | ZeTa = ", '{:>5}'.format(round((ZeTa_N+ZeTa_P)/2, 3)),
          " | maxRedEff = ", '{:>5}'.format(round((maxRedEff_N+maxRedEff_P)/2, 3)),
          )

    print("THEORETICAL :  ", '{:>35}'.format(" T = " + str(round(T,3))),
          " | ZT_PN = ", '{:>8}'.format(round(ZT_PN, 3)),
          " | Eta_Max = ", '{:>7}'.format(round(Eta_Max, 3)),
          " | EtaMaxEff = ", '{:>6}'.format(round(Eta_Max_Eff, 3)),
          " | Eta_Carnot = ", '{:>5}'.format(round(Eta_Carnot, 3)),
          )


for datanum in range(len(days)):
    print(" ")
    # print(" ")


    dfRaw = pd.read_csv('Data/Clean/' + days[datanum] + ".txt", skiprows=lambda x: logic(x))
    """ To take a portion of dataframe """

    df = dfRaw[int(len(dfRaw) * lowRange):int(len(dfRaw) * highRange)]
    try:
        df.drop(["MX1601_CH_11_[V]", "MX1601_CH_12_[V]", "MX1601_CH_13_[mA]", "MX1601_CH_14_[mA]",], axis=1, inplace=True )

    except:
        # print("No column dropped")

        print()

    index = []
    """ PLOT """

    # print_stats(df_Raw)

    """TO DROP part of data which is unstable, it takes out based on INDEX NOT TIME"""
    if   days[datanum]=="231010 - Energy balance with TEG_ARDUINO":
        df = df.drop(df.index[:600])
        df = df.drop(df.index[550:])
    elif days[datanum]=="231010 - With TEG_CATMAN":
        df = df.drop(df.index[3600:20000])
    elif days[datanum]=="231010 - Energy balance with space(No TEG)_ARDUINO":
        df = df.drop(df.index[0:1])
    elif days[datanum]=="231010 - With NO TEG_CATMAN":
        df = df.drop(df.index[:4000])

    """PLot on the basis of column number"""
    if days[datanum][-7:] == "ARDUINO":
        print(days[datanum])
        #     if datanum%2==0:
        col_num_ro_plot = [0,
                           1,
                           2,
                           3,
                           4,
                           4,
                           4,
                           4,
                           4,]
        # print(df.columns)
        df.rename(columns={
            "Time": "Time_[s]",
            " T_A[degC]": "T9_cold_out_TEGs_[degC]",
            " T_B[degC]": "T10_hot_out_TEGs_[degC]",
            " T_C[degC]": "T1_cold_in_TEGs_[degC]",
            " T_D[degC]": "T3_hot_in_TEGs_[degC]",
        }, inplace=True)
        print_stats(df)

        Time =              df["Time_[s]"]
        # print(df.columns)
        T9_cold_out_TEGs =  df["T9_cold_out_TEGs_[degC]"].tolist()
        T10_hot_out_TEGs =  df["T10_hot_out_TEGs_[degC]"].tolist()
        T1_cold_in_TEGs_ard =    df["T1_cold_in_TEGs_[degC]"].tolist()
        T3_hot_in_TEGs_ard =    df["T3_hot_in_TEGs_[degC]"].tolist()


        mean_T9_cold_out_TEGs = np.mean(T9_cold_out_TEGs)
        mean_T10_hot_out_TEGs = np.mean(T10_hot_out_TEGs)
        mean_T1_cold_in_TEGs_ard  = np.mean(T1_cold_in_TEGs_ard)
        mean_T3_hot_in_TEGs_ard  = np.mean(T3_hot_in_TEGs_ard)

        # print("mean_T9_cold_out_TEGs       ", mean_T9_cold_out_TEGs)
        # print("mean_T10_hot_out_TEGs       ", mean_T10_hot_out_TEGs)
        # print("mean_T1_cold_in_TEGs_ard    ", mean_T1_cold_in_TEGs_ard)
        # print("mean_T3_hot_in_TEGs_ard     ", mean_T3_hot_in_TEGs_ard)

    elif days[datanum][-6:] == "CATMAN":

        col_num_ro_plot = [0 ,
                           18,
                           19,
                           20,
                           21,
                           22,
                           23,
                           24,
                           25,
                           ]
        df.rename(columns={
            "Time1_[s]": "Time_[s]",
            "Time2_[s]": "Time_2_[s]",
            "Current_C_[]": "Current_C_[A]",
            "Power_C_[]": "Power_C_[W]",
            "Current_D_[]": "Current_D_[A]",
            "Power_D_[]": "Power_D_[W]",
            "Current_E_[]": "Current_E_[A]",
            "Power_E_[]": "Power_E_[W]",
            "Current_VarR_[]": "Current_VarR_[A]",
            "Power_VarR_[]": "Power_VarR_[W]",
            "DT_surf_A_K]": "DT_surf_A_[degC]",
            "DT_surf_B_[]": "DT_surf_B_[degC]",
            "DT_surf_C_[]": "DT_surf_C_[degC]",


            }, inplace=True)

        T5_cold_surf_B = df["T5_cold_surf_B_[degC]"].tolist()
        T8_cold_surf_C = df["T8_cold_surf_C_[degC]"].tolist()
        T2_hot_surf_C = df["T2_hot_surf_C_[degC]"].tolist()
        T4_hot_surf_A = df["T4_hot_surf_A_[degC]"].tolist()
        T3_hot_in_TEGs_cat = df["T3_hot_in_TEGs_[degC]"].tolist()
        T6_hot_surf_B = df["T6_hot_surf_B_[degC]"].tolist()
        T7_cold_surf_A = df["T7_cold_surf_A_[degC]"].tolist()
        T1_cold_in_TEGs_cat = df["T1_cold_in_TEGs_[degC]"].tolist()
        Power_D = df["Power_D_[W]"].tolist()
        Power_E = df["Power_E_[W]"].tolist()
        Power_VarR = df["Power_VarR_[W]"].tolist()
        DT_surf_A = df["DT_surf_A_[degC]"].tolist()
        DT_surf_B = df["DT_surf_B_[degC]"].tolist()
        DT_surf_C = df["DT_surf_C_[degC]"].tolist()

        Voltage_A = df["Voltage_10_A_[V]"].tolist()
        Voltage_B = df["Voltage_8.2_B_[V]"].tolist()
        Voltage_C = df["Voltage_9.09_C_[V]"].tolist()
        Voltage_D = df["Voltage_4.1_D_[V]"].tolist()
        Voltage_E = df["Voltage_4.1_E_[V]"].tolist()
        Voltage_VarR = df["Voltage_VarR_[V]"].tolist()
        Current_A = df["Current_A_[A]"].tolist()
        Power_A = df["Power_A_[W]"].tolist()
        Current_B = df["Current_B_[A]"].tolist()
        Power_B = df["Power_B_[W]"].tolist()
        Current_C = df["Current_C_[A]"].tolist()
        Power_C = df["Power_C_[W]"].tolist()
        Current_D = df["Current_D_[A]"].tolist()
        Power_D = df["Power_D_[W]"].tolist()
        Current_E = df["Current_E_[A]"].tolist()
        Power_E = df["Power_E_[W]"].tolist()
        Current_VarR = df["Current_VarR_[A]"].tolist()
        Power_VarR = df["Power_VarR_[W]"].tolist()

        mean_T5_cold_surf_B          = np.mean(T5_cold_surf_B)
        mean_T8_cold_surf_C          = np.mean(T8_cold_surf_C)
        mean_T2_hot_surf_C           = np.mean(T2_hot_surf_C)
        mean_T4_hot_surf_A           = np.mean(T4_hot_surf_A)
        mean_T3_hot_in_TEGs_cat      = np.mean(T3_hot_in_TEGs_cat)
        mean_T6_hot_surf_B           = np.mean(T6_hot_surf_B)
        mean_T7_cold_surf_A          = np.mean(T7_cold_surf_A)
        mean_T1_cold_in_TEGs_cat     = np.mean(T1_cold_in_TEGs_cat)
        mean_Power_D                 = np.mean(Power_D)
        mean_Power_E                 = np.mean(Power_E)
        mean_Power_VarR              = np.mean(Power_VarR)
        mean_DT_surf_A               = np.mean(DT_surf_A)
        mean_DT_surf_B               = np.mean(DT_surf_B)
        mean_DT_surf_C               = np.mean(DT_surf_C)

        mean_Voltage_A    = np.mean(Voltage_A)
        mean_Voltage_B    = np.mean(Voltage_B)
        mean_Voltage_C    = np.mean(Voltage_C)
        mean_Voltage_D    = np.mean(Voltage_D)
        mean_Voltage_E    = np.mean(Voltage_E)
        mean_Voltage_VarR       = np.mean(Voltage_VarR)
        mean_Current_A    = np.mean(Current_A)
        mean_Power_A  = np.mean(Power_A)
        mean_Current_B    = np.mean(Current_B)
        mean_Power_B  = np.mean(Power_B)
        mean_Current_C    = np.mean(Current_C)
        mean_Power_C  = np.mean(Power_C)
        mean_Current_D    = np.mean(Current_D)
        mean_Power_D  = np.mean(Power_D)
        mean_Current_E    = np.mean(Current_E)
        mean_Power_E  = np.mean(Power_E)
        mean_Current_VarR       = np.mean(Current_VarR)
        mean_Power_VarR     = np.mean(Power_VarR)


    col_list = df.columns
    data_to_plot_0 = df[col_list[0]]

    """_________________________________________________________________________"""

    mov_ave_data_to_plot_1 = mov_ave(df[df.columns[col_num_ro_plot[1]]].tolist(), nMovAve)
    mov_ave_data_to_plot_2 = mov_ave(df[df.columns[col_num_ro_plot[2]]].tolist(), nMovAve)
    mov_ave_data_to_plot_3 = mov_ave(df[df.columns[col_num_ro_plot[3]]].tolist(), nMovAve)
    mov_ave_data_to_plot_4 = mov_ave(df[df.columns[col_num_ro_plot[4]]].tolist(), nMovAve)
    mov_ave_data_to_plot_5 = mov_ave(df[df.columns[col_num_ro_plot[5]]].tolist(), nMovAve)
    mov_ave_data_to_plot_6 = mov_ave(df[df.columns[col_num_ro_plot[6]]].tolist(), nMovAve)
    mov_ave_data_to_plot_7 = mov_ave(df[df.columns[col_num_ro_plot[7]]].tolist(), nMovAve)
    mov_ave_data_to_plot_8 = mov_ave(df[df.columns[col_num_ro_plot[8]]].tolist(), nMovAve)

    data_label_to_plot_0 = df.columns[col_num_ro_plot[0]]
    data_label_to_plot_1 = df.columns[col_num_ro_plot[1]]
    data_label_to_plot_2 = df.columns[col_num_ro_plot[2]]
    data_label_to_plot_3 = df.columns[col_num_ro_plot[3]]
    data_label_to_plot_4 = df.columns[col_num_ro_plot[4]]
    data_label_to_plot_5 = df.columns[col_num_ro_plot[5]]
    data_label_to_plot_6 = df.columns[col_num_ro_plot[6]]
    data_label_to_plot_7 = df.columns[col_num_ro_plot[7]]
    data_label_to_plot_8 = df.columns[col_num_ro_plot[8]]

    plot_8(data_to_plot_0,
           mov_ave_data_to_plot_1,
           mov_ave_data_to_plot_2,
           mov_ave_data_to_plot_3,
           mov_ave_data_to_plot_4,
           mov_ave_data_to_plot_5,
           mov_ave_data_to_plot_6,
           mov_ave_data_to_plot_7,
           mov_ave_data_to_plot_8,

           [
               data_label_to_plot_0,
               data_label_to_plot_1,
               data_label_to_plot_2,
               data_label_to_plot_3,
               data_label_to_plot_4,
               data_label_to_plot_5,
               data_label_to_plot_6,
               data_label_to_plot_7,
               data_label_to_plot_8,
           ], days[datanum]

           )


    """  ______________________________________PRINT STATS _____________________________________________________"""

    # print_stats(df)

    """ ________________________________ CORRELATION PLOTS________________________________________________________ """
    # fCorr = plt.figure("CorrPlot Date " + days[datanum], figsize=(19, 15))
    # # plt.matshow(df.corr(), fignum=fCorr.number)
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=14)
    # plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=90)
    # plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    # plt.title('Correlation Matrix'+days[datanum], fontsize=16)
    """__________________________________________________________________________________________________________"""

    """ Calculating the ENERGY BALANCE  """
    """ NOTE: assumption is that glycol is 50% by weight .cTO BE CONFIRMED LATER  """

    """ TEST COOLPROP """
    """
    A call to the top-level function PropsSI can provide: temperature, pressure, density, heat capacity, internal energy, 
    enthalpy, entropy, viscosity and thermal conductivity. Hence, the available output keys are: T, P, D, C, U, H, S, V, L,
     Tmin and Tmax."""


    # print("Density of 50% (mass) ethylene glycol/water at 300 K, 101325 Pa:",
    #       PropsSI("V", "T", 275, "P", 101325, "INCOMP::MEG-0%"), "kg/m^3")
    # print("Viscosity of  50% (mass) ethylene glycol/water at 300 K, 101325 Pa:",
    #       PropsSI("V", "T", 275, "P", 101325, "INCOMP::MEG-50%"), "kg/m^3")

    substring_1 = "No TEG"
    substring_2 = "With TEG"
    circulating_fluid = "INCOMP::MEG-50%"

    if days[datanum].find("231020") != -1 and days[datanum].find("ContactHX") != -1:
        HZ_cold = 20 # HZ_cold =5.8 @T=15 & RPM= 4500
        HZ_hot = 5.4 # HZ_Hot =5.6   @T=53 & RPM= 1500

    if days[datanum].find("231020") != -1 and days[datanum].find("SpacedHX") != -1:
        HZ_cold = 5.8 # HZ_cold =5.8 @T=-1.7 & RPM= 4500
        HZ_hot = 5.6 # HZ_Hot =5.6   @T=53 & RPM= 1500

    if days[datanum].find("231020") != -1 and days[datanum].find("ZeroGap") != -1:
        HZ_cold = 5.6  # HZ_cold =5.8 @T=-1.7 & RPM= 4500
        HZ_hot = 5.4  # HZ_Hot =5.6   @T=53 & RPM= 1500

    if days[datanum].find("231020") != -1 and days[datanum].find("With TEG") != -1:
        HZ_cold = 5.8  # HZ_cold =5.8 @T=-1.7 & RPM= 4500
        HZ_hot = 5.4  # HZ_Hot =5.6   @T=53 & RPM= 1500

    if days[datanum].find("231020") != -1 and days[datanum].find("Test2") != -1:
        HZ_cold = 7.6  # HZ_cold =7.6 @T=0.2 & RPM= 3800
        HZ_hot = 6  # HZ_Hot =6   @T=60 & RPM= 1500

    try:

        flow_rate_cold = 0.307*HZ_cold/60*1000  # ml/sec
        flow_rate_hot = 0.307*HZ_hot/60*1000  # ml/sec

    except:
        print("Exception happended on HZ for cold & hot")
    # HZ_Hot =5.66   @T=53 & RPM= 1500

    if days[datanum].find("231010") != -1:
        """ 231010 Average Cold flow rate = 14.1 [ml/sec] Hot flow rate =3 2.5 [ml/sec] """
        flow_rate_cold = 14.1  # ml/sec
        flow_rate_hot = 32.5  # ml/sec


    try:

        T_C_in  = mean_T1_cold_in_TEGs_ard
        T_C_out = mean_T9_cold_out_TEGs
        T_H_in  = mean_T3_hot_in_TEGs_ard
        T_H_out = mean_T10_hot_out_TEGs
        t_flow_hot = (T_H_out + T_H_in) / 2
        t_flow_cold = (T_C_out + T_C_in) / 2

        m_dot_hot = ((PropsSI("D", "T", t_flow_hot + CtoK, "P", 101325, circulating_fluid)) * flow_rate_hot / 1e6)
        Q_hot = m_dot_hot * PropsSI("C", "T", t_flow_hot + CtoK, "P", 101325, circulating_fluid) * (
                    T_H_out - T_H_in)

        m_dot_cold = ((PropsSI("D", "T", t_flow_cold + CtoK, "P", 101325, circulating_fluid)) * flow_rate_cold / 1e6)
        Q_cold = m_dot_cold * PropsSI("C", "T", t_flow_cold + CtoK, "P", 101325, circulating_fluid) * (
                    T_C_out - T_C_in)
        if days[datanum][-6:] == "CATMAN":
            print("________________________________", "For data file:   ", days[datanum][:-7],
                  "_______________________________")
            print("EXPERIMENTAL : ", '{:>35}'.format(" T_in & T_Out  "),
                  " | Q_hot= ", '{:>9}'.format(round(Q_hot,4)),
                  " | Q_cold= ", '{:>9}'.format(round(Q_cold,4)),
                  " | W_ave(3TEG)= ", '{:>6}'.format(round(mean_Power_VarR*3,4)),
                  " | TEG eff", '{:>6}'.format(round((mean_Power_VarR * 3)/ (abs(Q_hot) - abs(Q_cold)), 4) ),
                  )

            """The total heat excanged is done with use of 3 TEG element"""
            T_C = (mean_T5_cold_surf_B + mean_T7_cold_surf_A + mean_T8_cold_surf_C)/3
            T_H = (mean_T2_hot_surf_C + mean_T4_hot_surf_A + mean_T6_hot_surf_B)/3
            DT = T_H - T_C
            # TEGEfficiencyEstimator(T_C, T_H)


    except:
        " "

    # elif days[datanum].find(substring_2) != -1:
    try:
        T_C_in  = mean_T1_cold_in_TEGs_cat
        T_C_out = mean_T7_cold_surf_A
        T_H_in  = mean_T3_hot_in_TEGs_cat
        T_H_out = mean_T2_hot_surf_C
        t_flow_hot = (T_H_out + T_H_in) / 2
        t_flow_cold = (T_C_out + T_C_in) / 2

        m_dot_hot = ((PropsSI("D", "T", t_flow_hot + CtoK, "P", 101325, circulating_fluid)) * flow_rate_hot / 1e6)
        Q_hot = m_dot_hot * PropsSI("C", "T", t_flow_hot + CtoK, "P", 101325, circulating_fluid) * (
                    T_C_out - T_C_in)

        m_dot_cold = ((PropsSI("D", "T", t_flow_cold + CtoK, "P", 101325, circulating_fluid)) * flow_rate_cold / 1e6)
        Q_cold = m_dot_cold * PropsSI("C", "T", t_flow_cold + CtoK, "P", 101325, circulating_fluid) * (
                    T_C_out - T_C_in)
        if days[datanum][-6:] == "CATMAN":
            # print("________________________________", "For data file:   ", days[datanum],
            #       "_______________________________")

            V = mean_Voltage_VarR
            I = mean_Current_VarR
            W = mean_Power_VarR
            T_C = (mean_T5_cold_surf_B + mean_T7_cold_surf_A + mean_T8_cold_surf_C)/3
            T_H = (mean_T2_hot_surf_C + mean_T4_hot_surf_A + mean_T6_hot_surf_B)/3
            DT = T_H - T_C




            print("EXPERIMENTAL : ", '{:>35}'.format("Based on T_in & T_surface"),
                  " | Q_hot= ", '{:>9}'.format(round(Q_hot,4)),
                  " | Q_cold= ", '{:>9}'.format(round(Q_cold,4)),
                  " | W_ave(3TEG)= ", '{:>6}'.format(round(mean_Power_VarR*3,4)),
                  " | TEG eff", '{:>6}'.format(round((mean_Power_VarR * 3)/ (abs(Q_hot) - abs(Q_cold)), 4) ),
                  )



        T_C_in  = mean_T8_cold_surf_C
        T_C_out = mean_T7_cold_surf_A
        T_H_in  = mean_T4_hot_surf_A
        T_H_out = mean_T2_hot_surf_C

        t_flow_hot = (T_H_out + T_H_in) / 2
        t_flow_cold = ( T_C_out + T_C_in) / 2

        m_dot_hot = ((PropsSI("D", "T", t_flow_hot + CtoK, "P", 101325, circulating_fluid)) * flow_rate_hot / 1e6)
        Q_hot = m_dot_hot * PropsSI("C", "T", t_flow_hot + CtoK, "P", 101325, circulating_fluid) * (
                    T_H_out - T_H_in)

        m_dot_cold = ((PropsSI("D", "T", t_flow_cold + CtoK, "P", 101325, circulating_fluid)) * flow_rate_cold / 1e6)
        Q_cold = m_dot_cold * PropsSI("C", "T", t_flow_cold + CtoK, "P", 101325, circulating_fluid) * (
                    T_C_out - T_C_in)
        if days[datanum][-6:] == "CATMAN":
            # print("________________________________", "For data file:   ", days[datanum],
            #       "_______________________________")

            V = mean_Voltage_VarR
            I = mean_Current_VarR
            W = mean_Power_VarR



            print("EXPERIMENTAL : ", '{:>35}'.format(" Based on T_surface"),
                  " | Q_hot= ", '{:>9}'.format(round(Q_hot,4)),
                  " | Q_cold= ", '{:>9}'.format(round(Q_cold,4)),
                  " | W_ave(3TEG)= ", '{:>6}'.format(round(mean_Power_VarR*3,4)),
                  " | TEG eff", '{:>6}'.format(round((mean_Power_VarR * 3)/ (abs(Q_hot) - abs(Q_cold)), 4) ),
                  )
            print("EXPERIMENTAL : ", '{:>35}'.format(" DT = " +str(round(DT, 3))),
                  " | V :    ", '{:>9}'.format(round(V, 3)),
                  " | I :     ", '{:>9}'.format(round(I, 3)),
                  " | W :    ", '{:>9}'.format(round(W, 3)),
                  )
            if days[datanum].find("With TEG") != -1:
                TEGEfficiencyEstimator(T_C, T_H)

    except:
        ""

    if days[datanum].find("CATMAN") != -1:
        print("##########______________****************************************____________############")

# plt.show()






