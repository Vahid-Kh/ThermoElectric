from CoolProp.CoolProp import PropsSI
import pandas as pd  # Dataframe library
from _functions import *

NumTEGInHX = 3

""" ___ CONSTANTS ___ """
step = 1
nMovAve = 30
nVarAve = 30
lowRange = 0
highRange = 1
printInterval = 1000
CtoK = + 273.15
circulating_fluid = "INCOMP::MEG-50%"


""" Choose data from a list of available data"""

""" must be ordered in a where data from Arduiono is 1st and CATMAN 2nd"""
days = [
    "231107 - DT20 to DT70 and Back - ARDUINORPS_1HZ",
    "231107 - Series, Parallel & Single - ARDUINORPS_1HZ",

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





for datanum in range(len(days)):
    print(" ")
    # print(" ")
    df = pd.read_csv('Data/Merged/' + days[datanum] + ".txt", skiprows=lambda x: logic(x))
    """ To take a portion of dataframe """



    RPS_Cold_mv = mov_ave(df["RPS_Cold"].tolist(),nMovAve)
    RPS_Hot_mv = mov_ave(df["RPS_Hot"].tolist(), nMovAve)

    flow_rate_cold_A = 0.307*np.array(RPS_Cold_mv)/60*1000  # ml/sec
    flow_rate_hot_A  = 0.307*np.array(RPS_Hot_mv)/60*1000  # ml/sec

    """ A ThermoElectric Efficiency Estimator scrip:
        1- Calculation based on datasheet estimate of Voltage & Current &  Calculation based on Zeta values
        2- Calculation of Zeta & max Efficiency based on PN junction semiconductor Seebeck effect
        """

    """  1- Calculation based on datasheet estimate of Voltage & Current &  Calculation based on Zeta values   """

    T5_cold_surf_B_A = np.array(df["T5_cold_surf_B_[degC]"])
    T8_cold_surf_C_A = np.array(df["T8_cold_surf_C_[degC]"])
    T7_cold_surf_A_A = np.array(df["T7_cold_surf_A_[degC]"])
    T2_hot_surf_C_A  = np.array(df["T2_hot_surf_C_[degC]"])
    T4_hot_surf_A_A  = np.array(df["T4_hot_surf_A_[degC]"])
    T6_hot_surf_B_A  = np.array(df["T6_hot_surf_B_[degC]"])

    T_cold_surf_A = (T5_cold_surf_B_A + T8_cold_surf_C_A + T7_cold_surf_A_A)/3
    T_hot_surf_A = (T2_hot_surf_C_A + T4_hot_surf_A_A + T6_hot_surf_B_A)/3

    """Temperature of TEG for its property calculations"""
    CtoK = 273.15
    T_C = np.array(T_cold_surf_A)
    T_H = np.array(T_hot_surf_A)

    T_ave = (T_H + T_C) / 2
    T=T_ave
    DT = T_H - T_C

    T1_cold_in_TEGs_mv = mov_ave(df["T1_cold_in_TEGs_[degC]"].tolist(), nMovAve)
    T3_hot_in_TEGs_mv = mov_ave(df["T3_hot_in_TEGs_[degC]"].tolist(), nMovAve)

    """TEMPLATES"""
    """Moving average"""
    # _mv = mov_ave(df[" "].tolist(), nMovAve)
    """Array """
    # _A = np.array(df[" "])

    """ Data sheet estimates """

    V_datasheet = 0.0473 * DT - 0.124
    I_datasheet = 0.0054 * DT + 0.1344

    W_datasheet = V_datasheet * I_datasheet



    T_A_Ard_A = np.array(df[" T_A[degC]"])
    T_B_Ard_A = np.array(df[" T_B[degC]"])
    T_C_Ard_A = np.array(df[" T_C[degC]"])
    T_D_Ard_A = np.array(df[" T_D[degC]"])

    T_C_in  = T_C_Ard_A
    T_C_out = T_A_Ard_A
    T_H_in  = T_D_Ard_A
    T_H_out = T_B_Ard_A

    t_flow_hot = (T_H_out + T_H_in) / 2
    t_flow_cold = (T_C_out + T_C_in) / 2

    """The total heat excanged is done with use of 3 TEG element"""

    m_dot_hot = ((PropsSI("D", "T", t_flow_hot + CtoK, "P", 101325, circulating_fluid)) * flow_rate_hot_A / 1e6)
    Q_hot = m_dot_hot * PropsSI("C", "T", t_flow_hot + CtoK, "P", 101325, circulating_fluid) * (T_H_out - T_H_in)

    m_dot_cold = ((PropsSI("D", "T", t_flow_cold + CtoK, "P", 101325, circulating_fluid)) * flow_rate_cold_A / 1e6)
    Q_cold = m_dot_cold * PropsSI("C", "T", t_flow_cold + CtoK, "P", 101325, circulating_fluid) * (T_C_out - T_C_in)

    """All the current is SUMMED UP and VOLTAGE is AVERAGED"""


    """Array """
    # _A = mov_ave(df[""].tolist(), nMovAve)
    Time = df["Time"].tolist()
    Voltage_A_A =  np.array(df["Voltage_10_A_[V]"])
    Voltage_B_A =  np.array(df["Voltage_8.2_B_[V]"])
    Voltage_C_A =  np.array(df["Voltage_9.09_C_[V]"])
    Voltage_D_A =  np.array(df["Voltage_4.1_D_[V]"])
    Voltage_E_A =  np.array(df["Voltage_4.1_E_[V]"])
    Voltage_V_A =  np.array(df["Voltage_VarR_[V]"])
    Current_A_A =  np.array(df["Current_A_[A]"])
    Current_B_A =  np.array(df["Current_B_[A]"])
    Current_C_A =  np.array(df["Current_C_[]"])
    Current_D_A =  np.array(df["Current_D_[]"])
    Current_E_A =  np.array(df["Current_E_[]"])
    Current_V_A =  np.array(df["Current_VarR_[]"])
    Power_A_A   =  np.array(df["Power_A_[W]"])
    Power_B_A   =  np.array(df["Power_B_[W]"])
    Power_C_A   =  np.array(df["Power_C_[]"])
    Power_D_A   =  np.array(df["Power_D_[]"])
    Power_E_A   =  np.array(df["Power_E_[]"])
    Power_V_A   =  np.array(df["Power_VarR_[]"])

    # _A = np.array(df[" "])

    V_imped_match = Voltage_V_A
    I_imped_match = Current_V_A
    W_Total = ((Power_A_A + Power_B_A + Power_C_A + Power_D_A + Power_E_A + Power_V_A)/NumTEGInHX).tolist()


    def replace_values(array, threshold, new_value):
        return [i if i <= threshold else new_value for i in array]


    # example usage


    TEG_eff = abs(W_Total/ ((abs(Q_hot) - abs(Q_cold))) )
    plot_2(Time, abs(Q_hot),abs(Q_cold),"    " , "   ")
    TEG_eff = replace_values(TEG_eff, 10,0)
    """THEORITICAL """
    CtoK = 273.15
    T_ave = (T_H + T_C) / 2
    T=T_ave

    """ Data sheet estimates """


    """  Calculation based on Zeta values """

    """ ppppppppp  for p-PbTe """
    S_P = .34987391 * T + 114.4786318
    # R_TEG = 1e-5 * (8e-6*T**2 - 0.007*T + 2.6118)  # Resistivity [Ohm * Meter]
    R_P = (-1e-8 * T ** 3 + 1e-5 * T ** 2 + 0.0017 * T + 0.6186)  # Resistivity [mOhm * cm]
    K_P = 8e-6 * T ** 2 - 0.007 * T + 2.6118  # Thermal Conductivity [W/m/K]

    """ nnnnnnnnn for n-PbTe """

    S_N = -0.293284981 * T + -78.40910427
    # R_TEG = 1e-5 * (8e-6*T**2 - 0.007*T + 2.6118)  # Resistivity [Ohm * Meter]
    R_N = (-2e-9 * T ** 3 + 8e-6 * T ** 2 + 0.001 * T + 0.2729)  # Resistivity [mOhm * cm]
    K_N = 1e-5 * T ** 2 - 0.0096 * T + 3.4338  # Thermal Conductivity [W/m/K]

    """ 2- Calculation of Zeta & max Efficiency based on PN junction semiconductor Seebeck effect """
    ZT_PN = ((S_P - S_N) ** 2 * (T_ave + CtoK)) / ((R_N * K_N) ** 0.5 + (R_P * K_P) ** 0.5) ** 2 / 1e7
    Eta_Max = (DT / T_H) * ((1 + ZT_PN) ** 2 - 1) / ((1 + ZT_PN) ** 2 + T_C / T_H)
    Eta_Max_Eff = (np.sqrt(1 + ZT_PN) - 1) / (np.sqrt(1 + ZT_PN) + 1)
    Eta_Carnot = (1 - ((273.15 + T_C) / (273.15 + T_H)))

    time, data_label_to_plot_0 = Time , "Time "
    mov_ave_data_to_plot_1,data_label_to_plot_1 = W_Total, "Power"
    mov_ave_data_to_plot_2,data_label_to_plot_2 = Eta_Max , "Eta_Max "
    mov_ave_data_to_plot_3,data_label_to_plot_3 = Eta_Max_Eff , "Eta_Max_Eff "
    mov_ave_data_to_plot_4,data_label_to_plot_4 = Eta_Carnot , "Eta_Carnot "
    mov_ave_data_to_plot_5,data_label_to_plot_5 = TEG_eff , "TEG_eff "
    mov_ave_data_to_plot_6,data_label_to_plot_6 = W_datasheet, " W_Datasheet"
    mov_ave_data_to_plot_7,data_label_to_plot_7 = Current_V_A, "I "




    plot_7(time,
           mov_ave_data_to_plot_1,
           mov_ave_data_to_plot_2,
           mov_ave_data_to_plot_3,
           mov_ave_data_to_plot_4,
           mov_ave_data_to_plot_5,
           mov_ave_data_to_plot_6,
           mov_ave_data_to_plot_7,

           [
               data_label_to_plot_0,
               data_label_to_plot_1,
               data_label_to_plot_2,
               data_label_to_plot_3,
               data_label_to_plot_4,
               data_label_to_plot_5,
               data_label_to_plot_6,
               data_label_to_plot_7,

           ], days[datanum ] + " Synchronization Test "

           )


    print("##########______________****************************************____________############")
plt.show()


"""List of variables in DataFrame"""

"""
    ["Time",
     " T_A[degC]",
     " T_B[degC]",
     " T_C[degC]",
     " T_D[degC]",
     "RPS_Cold",
     "RPS_Hot",
     "Voltage_10_A_[V]",
     "Shunt_10_A_[V]",
     "Voltage_8.2_B_[V]",
     "Shunt_8.2_B_[V]",
     "Voltage_9.09_C_[V]",
     "Shunt_9.09_C_[V]",
     "Voltage_4.1_D_[V]",
     "Shunt_4.1_D_[V]",
     "Voltage_4.1_E_[V]",
     "Shunt_4.1_E_[V]",
     "Voltage_VarR_[V]",
     "Shunt_VarR_[V]",
     "Time2_[s]",
     "T5_cold_surf_B_[degC]",
     "T8_cold_surf_C_[degC]",
     "T2_hot_surf_C_[degC]",
     "T4_hot_surf_A_[degC]",
     "T3_hot_in_TEGs_[degC]",
     "T6_hot_surf_B_[degC]",
     "T7_cold_surf_A_[degC]",
     "T1_cold_in_TEGs_[degC]",
     "Current_A_[A]",
     "Power_A_[W]",
     "Current_B_[A]",
     "Power_B_[W]",
     "Current_C_[]",
     "Power_C_[]",
     "Current_D_[]",
     "Power_D_[]",
     "Current_E_[]",
     "Power_E_[]",
     "Current_VarR_[]",
     "Power_VarR_[]",
     "DT_surf_A_K]",
     "DT_surf_B_[]",
     "DT_surf_C_[]", ]
"""





