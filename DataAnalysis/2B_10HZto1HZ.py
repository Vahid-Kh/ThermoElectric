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
    # "231107 - Series, Parallel & Single - CATMAN",  # Start time : 11:45:20.100
    # "231107 - DT20 to DT70 and Back - CATMAN",  # Start time: 12:49:18.933
    "240202 - Viessmann - Start to stop - Power",  # Start time: 12:56:56.000
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
    """ To take a portion of dataframe """

    df = dfRaw[int(len(dfRaw) * lowRange):int(len(dfRaw) * highRange)]

    try:
        df.drop(["Time_1_slow_sample_rate_[s]"], axis=1, inplace=True)
    except:
        try:
            df.drop(["Time2_[s]"], axis=1, inplace=True)
        except:
            print("No column dropped")

    """ PLOT """
    print_stats(df)
    col_list = df.columns


    # Assuming 'df' is your DataFrame and 'value' is the column you want to calculate the moving average
    for col_i in df.columns[1:]:
        df[col_i] = df[col_i].rolling(window=10).mean()

    """Since rolling average turns Nan for first 10 samples, (Rolling window)"""
    # Reducing data frequency from 10 samples/second to 1 sample/second
    # Assuming 'time' is the time column in your DataFrame
    df=df[10:]

    # Convert 'time' column to datetime format
    # Convert 'time_seconds' column to datetime format
    df['time'] = pd.to_datetime(df[df.columns[0]], unit='s')

    fn_str = days[datanum]
    start_time = "20" + str(fn_str[:2]) + "-" + str(fn_str[2:4])+ "-" + str(fn_str[4:6]) + " 12:28:30"

    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], unit='s',origin=start_time)
    print(df)

    # Resample the data to reduce the frequency from 10 samples per second to 1 sample per second
    df = df.set_index(df.columns[0]).resample('1S').mean().reset_index()

    col = (
        rd.random(),
        rd.random(),
        rd.random()
    )

    df.rename(columns={df.columns[0] : 'Time_1HZ'}, inplace=True)
    print(df)
    print(df.columns)
    df.drop(" time", inplace=True)
    # # Convert the dictionary into a DataFrame
    # df_1HZ = pd.DataFrame(data)
    # "___________________________________________________________________" \
    # "___________________________________________________________________" \
    # "___________________________________________________________________"
    df.to_csv("Data/Clean/" + str(days[datanum])  +  "RPS_1HZ.txt", index=False)
    # print(df_1HZ.head())
    print(df)
plt.show()

