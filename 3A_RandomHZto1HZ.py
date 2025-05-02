import pandas as pd  # Dataframe library
from _functions import *

""" ___ CONSTANTS ___ """
"""  DUE TO 10HZ TO 1 HZ      """
step = 1
nMovAve = 10  # NOT
nVarAve = 30
lowRange = 0
highRange = 1
printInterval = 1000


days = [
    # "231107 - Series, Parallel & Single - ARDUINO",
    # "231107 - DT20 to DT70 and Back - ARDUINO",
    # "231107 - Series, Parallel & Single - CATMAN",
    # "231107 - DT20 to DT70 and Back - CATMAN",
    # "240202 - Viessmann - Start to stop - Power",  # Start time: 12:56:56.000
    "240202 - Viessmann - Start to stop - Temp",

]

startTimeDataAquisition = [
    # " 11:45:20.100",
    # " 12:49:18.933",
    " 12:56:26"

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

    dfRaw = pd.read_csv('Data/Clean/' + days[datanum] + ".txt", skiprows=lambda x: logic(x))
    dfRaw = dfRaw.rename(columns={dfRaw.columns[0]: 'Time'})
    """ To take a portion of dataframe """

    # df = dfRaw[int(len(dfRaw) * lowRange):int(len(dfRaw) * highRange)]
    #

    df=dfRaw
    """ PLOT """
    # print_stats(df)
    col_list = df.columns

    fn_str = days[datanum]
    """Since rolling average turns Nan for first 10 samples, (Rolling window)"""
    # Reducing data frequency from 10 samples/second to 1 sample/second
    # Assuming 'time' is the time column in your DataFrame
    start_time = "20" + str(fn_str[:2]) + "-" + str(fn_str[2:4])+ "-" + str(fn_str[4:6]) + str(startTimeDataAquisition[datanum])

    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], unit='s',origin=start_time)

    # Convert 'time' column to datetime format
    # Convert 'time_seconds' column to datetime format

    df.set_index('Time', inplace=True)
    # df['time'] = pd.to_datetime(df[df.columns[0]], unit='s')
    df_resampled = df.resample('S').mean()

    # Use interpolate to fill the NaN values
    df_1HZ = df_resampled.interpolate(method='linear')


    col = (
        rd.random(),
        rd.random(),
        rd.random()
    )
    print(df_1HZ)



    # df_1HZ['Time'] = df_1HZ.index.tolist()
    df_1HZ.insert(0, 'Time_1HZ', df_1HZ.index.tolist())
    df_1HZ['Time_1HZ'] = df_1HZ['Time_1HZ'].dt.floor('S')
    # Convert the dictionary into a DataFrame
    print(df_1HZ)
    # "___________________________________________________________________" \
    # "___________________________________________________________________" \
    # "___________________________________________________________________"
    df_1HZ.to_csv("Data/Clean/" + str(days[datanum])  +  "RPS_1HZ.txt", index=False)
    # print(df_1HZ.head())


