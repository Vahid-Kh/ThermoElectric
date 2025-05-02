import pandas as pd
from datetime import datetime
# Sample time series data with 1-second frequency

from CoolProp.CoolProp import PropsSI
import pandas as pd  # Dataframe library
import numpy as np  # Scientific computing with nD object support
import random as rd
import math
from functions import *
from scipy.signal import butter, lfilter, freqz
from TDN import TDN, PSI
from scipy.signal import correlate

""" TEST COOLPROP """
print("*********** INCOMPRESSIBLE FLUID AND BRINES *****************")
print("Density of 50% (mass) ethylene glycol/water at 300 K, 101325 Pa:",
      PropsSI("V", "T", 275, "P", 101325, "INCOMP::MEG-0%"), "kg/m^3")
print("Viscosity of  50% (mass) ethylene glycol/water at 300 K, 101325 Pa:",
      PropsSI("V", "T", 275, "P", 101325, "INCOMP::MEG-50%"), "kg/m^3")

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

    "231010 - Energy balance with TEG_ARDUINO",
    "231010 - With TEG_CATMAN",

    # "231010 - Energy balance with space(No TEG)_ARDUINO",
    # "231010 - With NO TEG_CATMAN",

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




# def SyncSameLenghtDataFrames(df1,df2):
#     sum_time_diff_1 = 0
#     list_time_1 = df1["Time_[s]"].tolist()
#
#     for i_freq in range(len(list_time_1) - 1):
#         sum_time_diff_1 += (list_time_1[i_freq + 1] - list_time_1[i_freq])
#
#     sum_time_diff_2 = 0
#     list_time_2 = df2["Time_[s]"].tolist()
#     for i_freq in range(len(list_time_2) - 1):
#         sum_time_diff_2 += (list_time_2[i_freq + 1] - list_time_2[i_freq])
#
#
#     freq_df_1 = sum_time_diff_1 / len(list_time_1)
#     freq_df_2 = sum_time_diff_2 / len(list_time_2)
#
#     """Less frequent """
#
#
#     print(df_double_freq.head)
#     print(df2)
#     if freq_df_1/freq_df_2 >=1:
#         window_size = int(freq_df_1/freq_df_2)  # Adjust this for the desired window size
#         # Calculate the moving average for each column
#         moving_avg_df = df1.rolling(window=window_size).mean()
#     else:
#         1
#     print(freq_df_1,freq_df_2)



for datanum in range(len(days)):
    print("________________________________________________________________________________")
    print("For data file:   ", days[datanum])

    dfRaw = pd.read_csv('Data/Clean/' + days[datanum] + ".txt", skiprows=lambda x: logic(x))
    """ To take a portion of dataframe """

    df_Raw = dfRaw[int(len(dfRaw) * lowRange):int(len(dfRaw) * highRange)]

    index = []
    """ PLOT """

    print_stats(df_Raw)


# # Double the frequency by adding 0.5-second intervals
# df_double_freq = df.resample('0.5S').asfreq()
#
# # Interpolate the missing values using linear interpolation
# df_double_freq['value'] = df_double_freq['value'].interpolate(method='linear')
#
#

df = df_Raw

print(df)



def AddFreqDataFrame(df,freq_needed_mS):
    """Mili Sec to be added"""
    print("Please make sure the column here is time in seconds", df[df.columns[0]])

    # Set the window size for the moving average,

    # Double the frequency by adding 0.5-second intervals
    # Convert seconds to a pandas datetime object

    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]].apply(lambda x: x * 1000), unit='ms')
    print(df[df.columns[0]])
    # Format the timestamp to 'YYYY-MM-DD HH:MM:SS'
    df[df.columns[0]] = df[df.columns[0]].dt.strftime('%Y-%m-%d %H:%M:%S:%mS')
    print(df.head())
    print(df[df.columns[0]])
    print(str(freq_needed_mS/1000) + 'S')
    df_double_freq = df.resample(str(freq_needed_mS/1000) + 'S').asfreq()

    # Interpolate the missing values using linear interpolation
    df_double_freq['value'] = df_double_freq['value'].interpolate(method='linear')

    return  df



df_double_freq= AddFreqDataFrame(df,500)



# Print the original and interpolated data
print('Original Data:')
print(df)
print('\nDoubled Frequency Data with Interpolation:')
print(df_double_freq)




