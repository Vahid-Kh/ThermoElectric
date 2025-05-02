import pandas as pd  # Dataframe library
from _functions import *
from datetime import datetime

""" ___ CONSTANTS ___ """
step = 1
nMovAve = 1
nVarAve = 30

""" Choose data from a list of available data"""

""" must be ordered in a where data from Arduiono is 1st and CATMAN 2nd"""
days = [

    "240202 - Viessmann - Start to stop - PowerRPS_1HZ",

    # "",
    # "",

]


def logic(index):
    if index % step == 0:
        return False
    return True


def print_stats(dataframe):
    # Calculate the average, min, and max for each column
    stats_df = dataframe.agg(['min', 'mean', 'max']).transpose()
    try:
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
    except:
        print("Data contains non number")


for datanum in range(round(len(days))):
    print(days[datanum], " : ", 'Data/Merged/' + days[datanum] + ".txt")

    df = pd.read_csv('Data/Merged/' + days[datanum * 3] + ".txt", skiprows=lambda x: logic(x))

    print(df.columns[0])
    print(df.shape)

    # Assuming df1 and df2 are your dataframes and 'time' is your time column
    # df1.columns[0] = pd.to_datetime(df1.columns[0])  # Ensure the column is in datetime format
    """ From time format to seconds for easier handling """
    t0 = datetime.strptime("00:00:00", "%H:%M:%S")  # Time refrence for date conversion
    # t0 = datetime.strptime("00:00:00", "%H:%M:%S")  # Time refrence for date conversion

    # [print(*row, sep=', ') for row in df1.values.tolist()];
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming df is your dataframe
    plt.figure("CorrPlot for " + days[datanum * 3], figsize=(20, 50))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    # print(df_1_2_3.columns)
    #
    Col_Name_df = df.columns
    Col_to_plot = [0,
                   1,
                   2,
                   ]
    time = df[Col_Name_df[Col_to_plot[0]]]
    # var1 = df_1_2_3[Col_Name_df[Col_to_plot[1]]]
    # var2 = df_1_2_3[Col_Name_df[Col_to_plot[2]]]
    #
    # Labels = [Col_Name_df[Col_to_plot[0]],
    #           Col_Name_df[Col_to_plot[1]],
    #           Col_Name_df[Col_to_plot[2]],]
    #
    #
    # plot_2(time,var1,var2,Labels, days[datanum*3] + " Synchronization Test  " )

    col_num_ro_plot = [0,
                       12,
                       14,
                       16,
                       18,
                       20,
                       25,
                       26,
                       ]
    mov_ave_data_to_plot_1 = df[df.columns[col_num_ro_plot[1]]].tolist()
    mov_ave_data_to_plot_2 = df[df.columns[col_num_ro_plot[2]]].tolist()
    mov_ave_data_to_plot_3 = df[df.columns[col_num_ro_plot[3]]].tolist()
    mov_ave_data_to_plot_4 = df[df.columns[col_num_ro_plot[4]]].tolist()
    mov_ave_data_to_plot_5 = df[df.columns[col_num_ro_plot[5]]].tolist()
    mov_ave_data_to_plot_6 = df[df.columns[col_num_ro_plot[6]]].tolist()
    mov_ave_data_to_plot_7 = df[df.columns[col_num_ro_plot[7]]].tolist()

    data_label_to_plot_0 = df.columns[col_num_ro_plot[0]]
    data_label_to_plot_1 = df.columns[col_num_ro_plot[1]]
    data_label_to_plot_2 = df.columns[col_num_ro_plot[2]]
    data_label_to_plot_3 = df.columns[col_num_ro_plot[3]]
    data_label_to_plot_4 = df.columns[col_num_ro_plot[4]]
    data_label_to_plot_5 = df.columns[col_num_ro_plot[5]]
    data_label_to_plot_6 = df.columns[col_num_ro_plot[6]]
    data_label_to_plot_7 = df.columns[col_num_ro_plot[7]]

    plot_7_maxed(time,
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
                 ], days[datanum * 3] + " Synchronization Test "

                 )


    P_4TEG_B = df["Power_B[W]"]
    P_4TEG_C = df["Power_C[]"]
    P_2TEG_D = df["Power_D_[]"]
    P_2TEG_E = df["Power_E[]"]
    P_1TEG_V = df["Power_VarR[]"]




    Sens3 = np.array(df["Sensor_3"].tolist())  # Refrigerant inlet to the HE
    Sens6 = np.array(df["Sensor_6"].tolist())  # Refrigerant outlet from the HE
    Sens4 = np.array(df["Sensor_4"].tolist())  # Water outlet from the HE
    Sens5 = np.array(df["Sensor_5"].tolist())  # Water inlet to the HE
    Sens9 = np.array(df["Sensor_9"].tolist())  # Hot side plate temperature
    Sens10 =np.array(df["Sensor_10"].tolist())  # Cold side plate temperature
    DT_in = abs(Sens6 - Sens3)
    DT_out = abs(Sens5 - Sens4)
    LMDT = (DT_in - DT_out) / (np.log(DT_in) - np.log(DT_out))


    plot_7_maxed(time,
                 P_4TEG_B/4,
                 P_4TEG_C/4,
                 P_2TEG_D/2,
                 P_2TEG_E/2,
                 P_1TEG_V,
                 Sens9-Sens10,
                 LMDT,

                 [
                     "Time",
                     "P_TEG_B [W]",
                     "P_TEG_C [W]",
                     "P_TEG_D [W]",
                     "P_TEG_E [W]",
                     "P_TEG_Var [W]",
                     "Plate tempreture gradient",
                     "LMTD on water and refrigerant loops",
                 ], days[datanum * 3] + " Synchronization Test "

                 )
    plot_7_maxed(time,
                 P_4TEG_B/4,
                 P_2TEG_D/2,
                 P_1TEG_V/1,
                 DT_in,
                 DT_out,
                 Sens9-Sens10,
                 LMDT,

                 [
                     "Time",
                     "P_TEG_B [W]",
                     "P_TEG_D [W]",
                     "P_TEG_Var [W]",
                     "DT_in" ,
                     "DT_out",
                     "Plate tempreture gradient",
                     "LMTD on water and refrigerant loops",
                 ], days[datanum * 3] + " Synchronization Test "

                 )

    plot_7_maxed(time,
                 P_1TEG_V,
                 Sens3,
                 Sens6,
                 Sens4,
                 Sens5,
                 Sens9,
                 Sens10,
                 [   "Time",
                     "P_1_TEG_Var [W]",
                     "Refrigerant inlet to the HE [T]",
                     "Refrigerant outlet from the HE [T]",
                     "Water outlet from the HE [T]",
                     "Water inlet to the HE [T]",
                     "Hot side plate temperature",
                     "Cold side plate temperature",
                 ], days[datanum * 3] + " Synchronization Test "
                 )

    plot_3_maxed(time,
                 P_1TEG_V,
                 Sens9 - Sens10,
                 LMDT,
                 [   "Time",
                     "P_1_TEG_Var [W]",
                     "Plate tempreture gradient[K]",
                     "LMTD on water and refrigerant loops[K]",

                 ], days[datanum * 3] + " Synchronization Test "
                 )


    for i in range(len(df.columns)): print(i, "  ", df.columns[i])


plt.show()
