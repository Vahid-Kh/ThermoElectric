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

    "231107 - DT20 to DT70 and Back - ARDUINORPS_1HZ",
    "231107 - DT20 to DT70 and Back - CATMAN - HZ Mass FlowRPS_1HZ",
    "231107 - DT20 to DT70 and Back - CATMANRPS_1HZ",

    "231107 - Series, Parallel & Single - ARDUINORPS_1HZ",
    "231107 - Series, Parallel & Single - CATMAN - HZ Mass FlowRPS_1HZ",
    "231107 - Series, Parallel & Single - CATMANRPS_1HZ",
    # "",
    # "",
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

    df2 = pd.read_csv('Data/Clean/' + days[datanum * 3 + 1] + ".txt", skiprows=lambda x: logic(x))
    df2 = df2.rename(columns={df2.columns[0]: 'Time'})

    df3 = pd.read_csv('Data/Clean/' + days[datanum * 3 + 2] + ".txt", skiprows=lambda x: logic(x))
    df3 = df3.rename(columns={df3.columns[0]: 'Time'})

    print(df1.shape, df2.shape, df3.shape, )

    # Assuming df1 and df2 are your dataframes and 'time' is your time column
    # df1.columns[0] = pd.to_datetime(df1.columns[0])  # Ensure the column is in datetime format
    """ From time format to seconds for easier handling """
    t0 = datetime.strptime("00:00:00", "%H:%M:%S")  # Time refrence for date conversion
    # t0 = datetime.strptime("00:00:00", "%H:%M:%S")  # Time refrence for date conversion

    # [print(*row, sep=', ') for row in df1.values.tolist()];
    for i in range(df1.shape[0]):
        DT = datetime.strptime(str(df1.iloc[i][0]), "%Y-%m-%d %H:%M:%S")

        """!!!!!HERE WE START TIME FROM OUR FIRST DATA """
        if i == 0:
            t0 = DT
        df1.at[i, 'Time'] = int(round((DT - t0).total_seconds()))
    for i in df1.columns:
        df1 = df1.astype({i: 'float'})

    # [print(*row, sep=', ') for row in df2.values.tolist()];
    for i in range(df2.shape[0]):
        DT = datetime.strptime(str(df2.iloc[i][0]), "%Y-%m-%d %H:%M:%S")

        """!!!!!HERE WE START TIME FROM OUR FIRST DATA """
        if i == 0:
            t0 = DT
        df2.at[i, 'Time'] = round((DT - t0).total_seconds())

    for i in df1.columns:
        df1 = df1.astype({i: 'float'})

    for i in range(df3.shape[0]):
        DT = datetime.strptime(str(df3.iloc[i][0]), "%Y-%m-%d %H:%M:%S")

        """!!!!!HERE WE START TIME FROM OUR FIRST DATA """
        if i == 0:
            t0 = DT
        df3.at[i, 'Time'] = round((DT - t0).total_seconds())

    # print(df1,df2,df3)
    for i in df1.columns:
        df1 = df1.astype({i: 'float'})

    join_cols = 'Time'
    # df_1_2 = pd.merge(df1, df2[df2.duplicated(subset=join_cols, keep='first') == False], on=join_cols)
    # df_1_2_3 = pd.merge(df_1_2, df3[df3.duplicated(subset=join_cols, keep='first') == False], on=join_cols)
    df_1_2 = pd.merge(df1, df2, on=join_cols)
    df_1_2_3 = pd.merge(df_1_2, df3, on=join_cols)

    print_stats(df_1_2_3)
    print(df1.shape, df2.shape, df3.shape, df_1_2.shape, df_1_2_3.shape, )
    # [print(*row, sep=', ') for row in df_1_2_3.values.tolist()];
    print(df_1_2_3)
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming df is your dataframe
    plt.figure("CorrPlot for " + days[datanum * 3], figsize=(20, 50))
    correlation_matrix = df_1_2_3.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    # print(df_1_2_3.columns)
    #
    Col_Name_df = df_1_2_3.columns
    Col_to_plot = [0,
                   1,
                   2,
                   ]
    time = df_1_2_3[Col_Name_df[Col_to_plot[0]]]
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
                       1,
                       2,
                       3,
                       4,
                       21,
                       26,
                       26,
                       ]
    mov_ave_data_to_plot_1 = df_1_2_3[df_1_2_3.columns[col_num_ro_plot[1]]].tolist()
    mov_ave_data_to_plot_2 = df_1_2_3[df_1_2_3.columns[col_num_ro_plot[2]]].tolist()
    mov_ave_data_to_plot_3 = df_1_2_3[df_1_2_3.columns[col_num_ro_plot[3]]].tolist()
    mov_ave_data_to_plot_4 = df_1_2_3[df_1_2_3.columns[col_num_ro_plot[4]]].tolist()
    mov_ave_data_to_plot_5 = df_1_2_3[df_1_2_3.columns[col_num_ro_plot[5]]].tolist()
    mov_ave_data_to_plot_6 = df_1_2_3[df_1_2_3.columns[col_num_ro_plot[6]]].tolist()
    mov_ave_data_to_plot_7 = df_1_2_3[df_1_2_3.columns[col_num_ro_plot[7]]].tolist()

    data_label_to_plot_0 = df_1_2_3.columns[col_num_ro_plot[0]]
    data_label_to_plot_1 = df_1_2_3.columns[col_num_ro_plot[1]]
    data_label_to_plot_2 = df_1_2_3.columns[col_num_ro_plot[2]]
    data_label_to_plot_3 = df_1_2_3.columns[col_num_ro_plot[3]]
    data_label_to_plot_4 = df_1_2_3.columns[col_num_ro_plot[4]]
    data_label_to_plot_5 = df_1_2_3.columns[col_num_ro_plot[5]]
    data_label_to_plot_6 = df_1_2_3.columns[col_num_ro_plot[6]]
    data_label_to_plot_7 = df_1_2_3.columns[col_num_ro_plot[7]]

    # plot_7(time,
    #        mov_ave_data_to_plot_1,
    #        mov_ave_data_to_plot_2,
    #        mov_ave_data_to_plot_3,
    #        mov_ave_data_to_plot_4,
    #        mov_ave_data_to_plot_5,
    #        mov_ave_data_to_plot_6,
    #        mov_ave_data_to_plot_7,
    #
    #        [
    #            data_label_to_plot_0,
    #            data_label_to_plot_1,
    #            data_label_to_plot_2,
    #            data_label_to_plot_3,
    #            data_label_to_plot_4,
    #            data_label_to_plot_5,
    #            data_label_to_plot_6,
    #            data_label_to_plot_7,
    #        ], days[datanum * 3] + " Synchronization Test "
    #
    #        )
    plot_4(time,
           mov_ave_data_to_plot_1,
           mov_ave_data_to_plot_2,
           mov_ave_data_to_plot_3,
           mov_ave_data_to_plot_4,


           [
               data_label_to_plot_0,
               data_label_to_plot_1,
               data_label_to_plot_2,
               data_label_to_plot_3,
               data_label_to_plot_4,

           ], days[datanum * 3] + " Synchronization Test "

           )

    for i in df_1_2_3.columns: print(i)
    # df_1_2_3.to_csv("Data/Merged/" + days[datanum*3] + ".txt", index=False)
plt.show()



