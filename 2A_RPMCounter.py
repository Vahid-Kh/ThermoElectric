import pandas as pd  # Dataframe library
from _functions import *

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
    "231107 - Series, Parallel & Single - CATMAN - HZ Mass Flow",
    "231107 - DT20 to DT70 and Back - CATMAN - HZ Mass Flow",

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
    print(" ____________________- ", days[datanum], " _____________________-              ")
    # print(" ")

    dfRaw = pd.read_csv('Data/Clean/' + days[datanum] + ".txt", skiprows=lambda x: logic(x))
    """ To take a portion of dataframe """

    df = dfRaw[int(len(dfRaw) * lowRange):int(len(dfRaw) * highRange)]
    # df = df[:50000]

    try:
        df.drop(["Time_1_slow_sample_rate_[s]"], axis=1, inplace=True)

    except:
        print("No column dropped")

    """ PLOT """
    print(df.head)
    print(df.columns)
    print_stats(df)
    col_list = df.columns

    col_num_ro_plot = [0,
                       1,
                       2,
                       2,
                       2,
                       2,
                       2,
                       2,
                       2,
                       ]

    data_to_plot_0 = df[col_list[0]]

    data_to_plot_1 = df[df.columns[col_num_ro_plot[1]]].tolist()
    data_to_plot_2 = df[df.columns[col_num_ro_plot[2]]].tolist()
    data_to_plot_3 = df[df.columns[col_num_ro_plot[3]]].tolist()
    data_to_plot_4 = df[df.columns[col_num_ro_plot[4]]].tolist()
    data_to_plot_5 = df[df.columns[col_num_ro_plot[5]]].tolist()
    data_to_plot_6 = df[df.columns[col_num_ro_plot[6]]].tolist()
    data_to_plot_7 = df[df.columns[col_num_ro_plot[7]]].tolist()
    data_to_plot_8 = df[df.columns[col_num_ro_plot[8]]].tolist()

    mov_ave_data_to_plot_1 = mov_ave(data_to_plot_1, nMovAve)
    mov_ave_data_to_plot_2 = mov_ave(data_to_plot_2, nMovAve)
    mov_ave_data_to_plot_3 = mov_ave(data_to_plot_3, nMovAve)
    mov_ave_data_to_plot_4 = mov_ave(data_to_plot_4, nMovAve)
    mov_ave_data_to_plot_5 = mov_ave(data_to_plot_5, nMovAve)
    mov_ave_data_to_plot_6 = mov_ave(data_to_plot_6, nMovAve)
    mov_ave_data_to_plot_7 = mov_ave(data_to_plot_7, nMovAve)
    mov_ave_data_to_plot_8 = mov_ave(data_to_plot_8, nMovAve)

    data_label_to_plot_0 = df.columns[col_num_ro_plot[0]]
    data_label_to_plot_1 = df.columns[col_num_ro_plot[1]]
    data_label_to_plot_2 = df.columns[col_num_ro_plot[2]]
    data_label_to_plot_3 = df.columns[col_num_ro_plot[3]]
    data_label_to_plot_4 = df.columns[col_num_ro_plot[4]]
    data_label_to_plot_5 = df.columns[col_num_ro_plot[5]]
    data_label_to_plot_6 = df.columns[col_num_ro_plot[6]]
    data_label_to_plot_7 = df.columns[col_num_ro_plot[7]]
    data_label_to_plot_8 = df.columns[col_num_ro_plot[8]]

    # plot_8(data_to_plot_0,
    #        mov_ave_data_to_plot_1,
    #        mov_ave_data_to_plot_2,
    #        mov_ave_data_to_plot_3,
    #        mov_ave_data_to_plot_4,
    #        mov_ave_data_to_plot_5,
    #        mov_ave_data_to_plot_6,
    #        mov_ave_data_to_plot_7,
    #        mov_ave_data_to_plot_8,
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
    #            data_label_to_plot_8,
    #        ], days[datanum]
    #
    #        )

    label = [data_label_to_plot_0,
             data_label_to_plot_1,
             data_label_to_plot_2,
             data_label_to_plot_3,
             data_label_to_plot_4,
             data_label_to_plot_5,
             data_label_to_plot_6,
             data_label_to_plot_7,
             data_label_to_plot_8, ]
    nMovAve = 1
    col = (
        rd.random(),
        rd.random(),
        rd.random()
    )


    def Count_RPM(highfrqTime, highFreq, batchsize):
        count = 0
        lowFreqCount = []
        lowFreqLen = round(len(highFreq) / batchsize)
        count_tot = 0
        recTimeCount = [0]
        lowFreqTime = []
        for ii in range(lowFreqLen - 1):
            start_time = ii * batchsize
            end_time = (ii + 1) * batchsize
            # print(start_time, end_time)
            count = 0
            timeStep = highfrqTime[1]-highfrqTime[0]
            for i in range(start_time, end_time):
                try:
                    if (+ (highFreq[i + 1]-highFreq[i + 0])
                        + (highFreq[i + 2]-highFreq[i + 1])
                        + (highFreq[i + 3]-highFreq[i + 2])
                        + (highFreq[i + 4]-highFreq[i + 3])
                        + (highFreq[i + 5]-highFreq[i + 4])
                        + (highFreq[i + 6]-highFreq[i + 5])
                    ) < -0.15:

                        # print(highfrqTime[i] - recTimeCount[-1],timeStep * 500)
                        if highfrqTime[i] - recTimeCount[-1] > timeStep * 100:
                            count += 1
                            # print("TIME RECOREDED: ", highfrqTime[i], "i , i+1, i+2, i+3 ", highFreq[i], highFreq[i + 1], highFreq[i + 2], highFreq[i + 3])

                            count_tot += 1
                            recTimeCount.append(highfrqTime[i])
                except:
                    print("EXCEPTION HAPPEND")

            lowFreqTime.append(highfrqTime[i-round(batchsize/2)])
            lowFreqCount.append(count)

        return lowFreqTime, lowFreqCount



    # plt.figure(days[datanum]+data_label_to_plot_1)
    # plt.plot(data_to_plot_0, var1)
    print(days[datanum]+data_label_to_plot_1)
    print(Count_RPM(data_to_plot_0, data_to_plot_1, 4800))  # Output: 2

    # plt.figure(days[datanum]+data_label_to_plot_2)
    # plt.plot(data_to_plot_0, var2)

    print(days[datanum]+data_label_to_plot_2)
    print(Count_RPM(data_to_plot_0, data_to_plot_2, 4800))  # Output: 2

    data = {'Time_1HZ': Count_RPM(data_to_plot_0, data_to_plot_1, 4800)[0],'RPS_Cold': Count_RPM(data_to_plot_0, data_to_plot_1, 4800)[1], 'RPS_Hot': Count_RPM(data_to_plot_0, data_to_plot_2, 4800)[1]}

    # Convert the dictionary into a DataFrame
    df_1HZ = pd.DataFrame(data)
    fn_str = days[datanum]
    startTimeDataAquisition = [
        " 11:45:20.100",
        " 12:49:18.933",

    ]

    start_time = "20" + str(fn_str[:2]) + "-" + str(fn_str[2:4])+ "-" + str(fn_str[4:6]) + str(startTimeDataAquisition[datanum])

    df_1HZ[df_1HZ.columns[0]] = pd.to_datetime(df_1HZ[df_1HZ.columns[0]], unit='s',origin=start_time)

    df_1HZ['Time_1HZ'] = df_1HZ['Time_1HZ'].dt.floor('S')
    "___________________________________________________________________" \
    "___________________________________________________________________" \
    "___________________________________________________________________"
    df_1HZ.to_csv("Data/Clean/" + str(days[datanum])  +  "RPS_1HZ.txt", index=False)
    print(df_1HZ.head())
plt.show()


