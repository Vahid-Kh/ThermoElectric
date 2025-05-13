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
    "240202 - Viessmann - Start to stop - TempRPS_1HZ",
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


for datanum in range(round(len(days) / 3)):
    print(days[datanum], " : ", 'Data/Clean/' + days[datanum * 3] + ".txt")

    df1 = pd.read_csv('Data/Clean/' + days[datanum * 3] + ".txt", skiprows=lambda x: logic(x))
    df1.rename(columns={df1.columns[0]: 'Time'}, inplace=True)
    print(df1.columns[0])
    df2 = pd.read_csv('Data/Clean/' + days[datanum * 3 + 1] + ".txt", skiprows=lambda x: logic(x))
    df2 = df2.rename(columns={df2.columns[0]: 'Time'})
    print(df2.columns[0])
    try:
        df1.drop(["time"], axis=1, inplace=True)
    except:
        df2.drop(["time"], axis=1, inplace=True)

    print(df1.shape, df2.shape, )

    # Assuming df1 and df2 are your dataframes and 'time' is your time column
    # df1.columns[0] = pd.to_datetime(df1.columns[0])  # Ensure the column is in datetime format
    """ From time format to seconds for easier handling """
    t0 = datetime.strptime("00:00:00", "%H:%M:%S")  # Time refrence for date conversion
    # t0 = datetime.strptime("00:00:00", "%H:%M:%S")  # Time refrence for date conversion

    # [print(*row, sep=', ') for row in df1.values.tolist()];
    for i in range(df1.shape[0]):
        try:
            DT = datetime.strptime(str(df1.iloc[i][0]), "%Y-%m-%d %H:%M:%S")
        except:
            DT = datetime.strptime(str(df1.iloc[i][0]), "%Y-%m-%d %H.%M.%S")

        """!!!!!HERE WE START TIME FROM OUR FIRST DATA """
        if i == 0:
            t0 = DT
        df1.at[i, 'Time'] = int(round((DT - t0).total_seconds()))
    for i in df1.columns:
        df1 = df1.astype({i: 'float'})

    # [print(*row, sep=', ') for row in df2.values.tolist()];
    for i in range(df2.shape[0]):

        try:
            DT = datetime.strptime(str(df2.iloc[i][0]), "%Y-%m-%d %H:%M:%S")
        except:
            DT = datetime.strptime(str(df2.iloc[i][0]), "%Y-%m-%d %H.%M.%S")

        """!!!!!HERE WE START TIME FROM OUR FIRST DATA """
        if i == 0:
            t0 = t0
        df2.at[i, 'Time'] = round((DT - t0).total_seconds())

    for i in df1.columns:
        df1 = df1.astype({i: 'float'})

    # print(df1,df2,df3)
    for i in df1.columns:
        df1 = df1.astype({i: 'float'})

    join_cols = 'Time'
    # df_1_2 = pd.merge(df1, df2[df2.duplicated(subset=join_cols, keep='first') == False], on=join_cols)
    # df_1_2_3 = pd.merge(df_1_2, df3[df3.duplicated(subset=join_cols, keep='first') == False], on=join_cols)
    df_1_2 = pd.merge(df1, df2, on=join_cols)

    print(df1.shape, df2.shape, df_1_2.shape)
    # [print(*row, sep=', ') for row in df_1_2_3.values.tolist()];

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming df is your dataframe
    plt.figure("CorrPlot for " + days[datanum * 3], figsize=(20, 50))
    correlation_matrix = df_1_2.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    # print(df_1_2_3.columns)
    #
    Col_Name_df = df_1_2.columns
    Col_to_plot = [0,
                   1,
                   2,
                   ]
    time = df_1_2[Col_Name_df[Col_to_plot[0]]]
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
    mov_ave_data_to_plot_1 = df_1_2[df_1_2.columns[col_num_ro_plot[1]]].tolist()
    mov_ave_data_to_plot_2 = df_1_2[df_1_2.columns[col_num_ro_plot[2]]].tolist()
    mov_ave_data_to_plot_3 = df_1_2[df_1_2.columns[col_num_ro_plot[3]]].tolist()
    mov_ave_data_to_plot_4 = df_1_2[df_1_2.columns[col_num_ro_plot[4]]].tolist()
    mov_ave_data_to_plot_5 = df_1_2[df_1_2.columns[col_num_ro_plot[5]]].tolist()
    mov_ave_data_to_plot_6 = df_1_2[df_1_2.columns[col_num_ro_plot[6]]].tolist()
    mov_ave_data_to_plot_7 = df_1_2[df_1_2.columns[col_num_ro_plot[7]]].tolist()

    data_label_to_plot_0 = df_1_2.columns[col_num_ro_plot[0]]
    data_label_to_plot_1 = df_1_2.columns[col_num_ro_plot[1]]
    data_label_to_plot_2 = df_1_2.columns[col_num_ro_plot[2]]
    data_label_to_plot_3 = df_1_2.columns[col_num_ro_plot[3]]
    data_label_to_plot_4 = df_1_2.columns[col_num_ro_plot[4]]
    data_label_to_plot_5 = df_1_2.columns[col_num_ro_plot[5]]
    data_label_to_plot_6 = df_1_2.columns[col_num_ro_plot[6]]
    data_label_to_plot_7 = df_1_2.columns[col_num_ro_plot[7]]

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

    Sens3 = np.array(df_1_2[df_1_2.columns[21]].tolist())
    Sens6 = np.array(df_1_2[df_1_2.columns[24]].tolist())
    Sens4 = np.array(df_1_2[df_1_2.columns[22]].tolist())
    Sens5 = np.array(df_1_2[df_1_2.columns[23]].tolist())
    DT_in = abs(Sens6 - Sens3)
    DT_out = abs(Sens5 - Sens4)
    LMDT = (DT_in - DT_out) / (np.log(DT_in) - np.log(DT_out))

    np.array(mov_ave_data_to_plot_6) - np.array(mov_ave_data_to_plot_7)
    plot_7_maxed(time,
                 mov_ave_data_to_plot_1,
                 mov_ave_data_to_plot_2,
                 mov_ave_data_to_plot_3,
                 mov_ave_data_to_plot_4,
                 mov_ave_data_to_plot_5,
                 np.array(mov_ave_data_to_plot_6) - np.array(mov_ave_data_to_plot_7),
                 LMDT,

                 [
                     "Time",
                     "P_4TEG_B [W]",
                     "P_4TEG_C [W]",
                     "P_2TEG_D [W]",
                     "P_2TEG_E [W]",
                     "P_1TEG_Var [W]",
                     "Plate tempreture gradient",
                     "LMTD on water and refrigerant loops",
                 ], days[datanum * 3] + " Synchronization Test "

                 )
    plot_7_maxed(time,
                 mov_ave_data_to_plot_1,
                 mov_ave_data_to_plot_3,
                 mov_ave_data_to_plot_5,
                 DT_in,
                 DT_out,
                 np.array(mov_ave_data_to_plot_6) - np.array(mov_ave_data_to_plot_7),
                 LMDT,

                 [
                     "Time",
                     "P_4_TEG_B [W]",
                     "P_2_TEG_D [W]",
                     "P_1_TEG_Var [W]",
                     "DT_in" ,
                     "DT_out",
                     "Plate tempreture gradient",
                     "LMTD on water and refrigerant loops",
                 ], days[datanum * 3] + " Synchronization Test "

                 )
    # plot_4(time,
    #        mov_ave_data_to_plot_1,
    #        mov_ave_data_to_plot_2,
    #        mov_ave_data_to_plot_3,
    #        mov_ave_data_to_plot_4,
    #
    #
    #        [
    #            data_label_to_plot_0,
    #            data_label_to_plot_1
    #            data_label_to_plot_2
    #            data_label_to_plot_3
    #            data_label_to_plot_4
    #
    #        ], days[datanum * 3] + " Synchronization Test "
    #
    #        )

    for i in range(len(df_1_2.columns)): print(i, "  ", df_1_2.columns[i])

    df_1_2.to_csv("Data/Merged/" + days[datanum * 3] + ".txt", index=False)

plt.show()
