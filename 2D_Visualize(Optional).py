
import pandas as pd                 # Dataframe library
from _functions import *

""" ___ CONSTANTS ___ """
step = 1
nMovAve = 1
nVarAve = 30
lowRange = 0
highRange = 1
printInterval = 1000
R = "R134a"

""" Choose data from a list of available data"""

days = [
    # "231010 - Energy balance with TEG_ARDUINO",
    # "231010 - With TEG_CATMAN",
    # "231010 - Energy balance with space(No TEG)_ARDUINO",
    # "231010 - With NO TEG_CATMAN",


    ]


def logic(index):
    if index % step == 0:
        return False
    return True

def print_stats(dataframe):
    # Calculate the average, min, and max for each column
    stats_df = dataframe.agg([ 'min','mean', 'max']).transpose()

    # Print the stats in a tabular format
    # print("num\t\tAverage\t\tMin\t\tMax")
    print("="*80)
    print("#", '{:>3}'.format("Num"),
          " | ", '{:>25}'.format("Column"),
          " | ", '{:>10}'.format('min'),
          " | ", '{:>10}'.format('mean'),
          " | ", '{:>10}'.format('max'),

          )
    print("_"*80)
    i_col_counter=0
    for index, row in stats_df.iterrows():

        # print(i_col_counter ,f"{index}\t\t{row['mean']:.2f}\t\t{row['min']}\t\t{row['max']}")

        print("#", '{:>3}'.format(i_col_counter),
              " | ", '{:>25}'.format(index),
              " | ", '{:>10}'.format(str(round(row['min'], 5))),
              " | ", '{:>10}'.format(str(round(row['mean'], 5))),
              " | ", '{:>10}'.format(str(round(row['max'], 5))),

              )
        i_col_counter+=1



for datanum in range(len(days)):
    print("________________________________________________________________________________")
    print("For data file:   ", days[datanum])

    dfRaw = pd.read_csv('Data/Clean/' + days[datanum] + ".txt", skiprows=lambda x: logic(x))
    """ To take a portion of dataframe """

    df_Raw = dfRaw[int(len(dfRaw)*lowRange):int(len(dfRaw)*highRange)]


    index = []
    """ PLOT """

    print_stats(df_Raw)
    col_list = df_Raw.columns


    """PLot on the basis of column number"""
    if days[datanum][-7:]=="ARDUINO":
        col_num_ro_plot = [0,
                           1,
                           2,
                           3,
                           4,
                           4,
                           4,
                           4,
                           ]
    else:
        col_num_ro_plot= [0 ,
                          22,
                          25 ,
                          38,
                          39,
                          40,
                          33,
                          35,
                          ]
    data_to_plot_0 = df_Raw[col_list[0]]
    mov_ave_data_to_plot_1 = mov_ave(df_Raw[df_Raw.columns[col_num_ro_plot[1]]].tolist(),nMovAve)
    mov_ave_data_to_plot_2 = mov_ave(df_Raw[df_Raw.columns[col_num_ro_plot[2]]].tolist(),nMovAve)
    mov_ave_data_to_plot_3 = mov_ave(df_Raw[df_Raw.columns[col_num_ro_plot[3]]].tolist(),nMovAve)
    mov_ave_data_to_plot_4 = mov_ave(df_Raw[df_Raw.columns[col_num_ro_plot[4]]].tolist(),nMovAve)
    mov_ave_data_to_plot_5 = mov_ave(df_Raw[df_Raw.columns[col_num_ro_plot[5]]].tolist(),nMovAve)
    mov_ave_data_to_plot_6 = mov_ave(df_Raw[df_Raw.columns[col_num_ro_plot[6]]].tolist(),nMovAve)
    mov_ave_data_to_plot_7 = mov_ave(df_Raw[df_Raw.columns[col_num_ro_plot[7]]].tolist(),nMovAve)

    data_label_to_plot_0 = df_Raw.columns[col_num_ro_plot[0]]
    data_label_to_plot_1 = df_Raw.columns[col_num_ro_plot[1]]
    data_label_to_plot_2 = df_Raw.columns[col_num_ro_plot[2]]
    data_label_to_plot_3 = df_Raw.columns[col_num_ro_plot[3]]
    data_label_to_plot_4 = df_Raw.columns[col_num_ro_plot[4]]
    data_label_to_plot_5 = df_Raw.columns[col_num_ro_plot[5]]
    data_label_to_plot_6 = df_Raw.columns[col_num_ro_plot[6]]
    data_label_to_plot_7 = df_Raw.columns[col_num_ro_plot[7]]

    # plot_4_maxed(data_to_plot_0,
    #              mov_ave_data_to_plot_1,
    #              mov_ave_data_to_plot_2,
    #              mov_ave_data_to_plot_3,
    #              mov_ave_data_to_plot_4,
    #              ["Time",
    #               data_label_to_plot_1,
    #               data_label_to_plot_2,
    #               data_label_to_plot_3,
    #               data_label_to_plot_4, ], days[datanum]
    #
    #              )

    plot_7(data_to_plot_0,
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



                  ], days[datanum]

                 )

    # plot_2_maxed(data_to_plot_0,
    #              mov_ave_data_to_plot_1,
    #              mov_ave_data_to_plot_2,
    #
    #              ["Time",
    #               data_label_to_plot_1,
    #               data_label_to_plot_2,
    #
    #               ], days[datanum]
    #
    #              )

    """ CORRELATION PLOTS """
    # fCorr = plt.figure("CorrPlot Date " + days[datanum], figsize=(19, 15))
    # plt.matshow(df_Raw.corr(), fignum=fCorr.number)
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=14)
    # plt.xticks(range(df_Raw.select_dtypes(['number']).shape[1]), df_Raw.select_dtypes(['number']).columns, fontsize=14, rotation=90)
    # plt.yticks(range(df_Raw.select_dtypes(['number']).shape[1]), df_Raw.select_dtypes(['number']).columns, fontsize=14)
    #
    # plt.title('Correlation Matrix'+days[datanum], fontsize=16)
    # #########################################################


plt.show()









