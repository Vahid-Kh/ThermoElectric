# encoding=utf8

import pandas as pd  # Dataframe library
import re
from datetime import datetime
import numpy as np  # Scientific computing with nD object support
import matplotlib.pyplot as plt
# from functions import mov_ave
# import seaborn as sns
from pandas.core.dtypes.common import is_string_dtype

"""@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"""

""" ALWAYS BE CAREFUL OF HEADER OF DATA
Export data in the Export tab with ACII format then remember to change file extension to txt

REMEMBER ALL UNCLOSED CHANNELS, THROUGH CATMAN SOFTWARE USING THE RELAY, WILL RECORD DATA AS ZERO AND THE HEADER REMAINS WHICH IS A WASTE
IF RECORDING DATA WITH WITH DIFFERENT FREQENCY /ACQUISITION TIME, REMEMBER TO EXTRACT SEPARATELY SO WOULD NOT MESS UP, FROM THE HIGHER FREQUENCY TAKE A MOVING AVERAGE AND MATCH THE NUMBER SOMEHOW 


Time1_[s]	Voltage_10_A_[V]	Shunt_10_A_[V]	Voltage_8.2_B_[V]	Shunt_8.2_B_[V]	Voltage_9.09_C_[V]	Shunt_9.09_C_[V]	Voltage_4.1_D_[V]	Shunt_4.1_D_[V]	Voltage_4.1_E_[V]	Shunt_4.1_E_[V]	Voltage_VarR_[V]	Shunt_VarR_[V]	Time2_[s]	T5_cold_surf_B_[degC]	T8_cold_surf_C_[degC]	T2_hot_surf_C_[degC]	T4_hot_surf_A_[degC]	T3_hot_in_TEGs_[degC]	T6_hot_surf_B_[degC]	T7_cold_surf_A_[degC]	T1_cold_in_TEGs_[degC]	Current_A_[A]	Power_A_[W]	Current_B_[A]	Power_B_[W]	Current_C_[]	Power_C_[]	Current_D_[]	Power_D_[]	Current_E_[]	Power_E_[]	Current_VarR_[]	Power_VarR_[]	DT_surf_A_K]	DT_surf_B_[]	DT_surf_C_[]
0	-0,000022	0,00014	-0,00036	-0,00030	-0,00039	-0,00027	1,576	0,3091	1,529	0,3003	1,221	-0,3865	0	9,834	8,452	49,90	52,37	57,79	51,88	9,658	3,516	0,00014	3,02e-9	0,00030	1,08e-7	0,00027	1,08e-7	0,3091	0,4871	0,3003	0,4592	0,3865	0,4717	42,71	42,05	41,45	


Time_1_slow_sample_rate_[s]	Time_1_fast_sample_rate_[s]	HZ_Flow_Cold_[V]	HZ_Flow_Hot_[V]


"""
"""@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"""

"""________________________________________________________________________________"""

data_file = [
    # "231010 - With NO TEG_CATMAN",
    # "231010 - With TEG_CATMAN",
    #
    # "231020_Test2_nrows12840_CATMAN",
    # "231020_NO TEG_ContactHX_nrows44276_CATMAN",
    # "231020_No TEG_SpacedHX_nrows6418_CATMAN",
    # "231020_ZeroGap_nrows2532_CATMAN",
    # "231020_With TEG_nrows8017_CATMAN",
    # "231107 - DT20 to DT70 and Back - CATMAN - HZ Mass Flow"
    # "231107 - Series, Parallel & Single - CATMAN - HZ Mass Flow"
    # "231107 - DT20 to DT70 and Back - CATMAN",
    # "231107 - Series, Parallel & Single - CATMAN",
    # "240202 - Viessmann - Start to stop - Power",
    "240202 - Viessmann - Start to stop - Temp",
    # "",
    # "",
    # "",
    # "",
    # "",
    # "",
    # "",

]

def replace_comma_with_period(input_file, output_file):
    try:
        # Read the content from the input file
        with open(input_file, 'r') as file:
            content = file.read()
        # Replace commas with periods
        modified_content = content.replace(',', '.')
        # Write the modified content to the output file
        with open(output_file, 'w') as file:
            file.write(modified_content)
        print('Replacement completed. Modified content saved to', output_file)

    except FileNotFoundError:
        print('Input file not found.')

def convert_space_to_comma(input_file, output_file):
    try:
        # Read the input file and process each line
        with open(input_file, 'r') as input_fp:
            lines = input_fp.readlines()
            # Process each line and write to the output file
            with open(output_file, 'w') as output_fp:
                for line in lines:
                    # Split by spaces and join with commas
                    comma_separated_line = ','.join(line.strip().split())
                    output_fp.write(comma_separated_line + '\n')
        print("Conversion successful. Output written to", output_file)

    except FileNotFoundError:
        print("Input file not found.")
    except Exception as e:
        print("An error occurred:", str(e))

def read_txt_file(filepath):
    try:
        # Read the text file with comma-separated data using pandas
        df = pd.read_csv(filepath)
        return df

    except FileNotFoundError:
        print("File not found.")
    except pd.errors.ParserError as pe:
        print("Error occurred while parsing the file:", str(pe))
    except Exception as e:
        print("An error occurred:", str(e))



# Display the data

for data_file_i in data_file:
    print(data_file_i)
    txt_comma_decimal_space_sep = 'Data/Raw/' + data_file_i + ".txt"  # Adjust to the actual file path
    txt_dot_decimal_space_sep = 'Data/Temp/' + data_file_i + ".txt" # Adjust to the desired output file path
    txt_dot_decimal_comma_sep = 'Data/Clean/' + data_file_i +  ".txt"  # Adjust to the desired output file path


    with open(txt_comma_decimal_space_sep, "r") as grilled_cheese:
        lines = grilled_cheese.readlines()
        # print(lines)

    # Replace commas with periods and save to the output file
    replace_comma_with_period(txt_comma_decimal_space_sep, txt_dot_decimal_space_sep)

    # Convert space-separated data to comma-separated data
    convert_space_to_comma(txt_dot_decimal_space_sep, txt_dot_decimal_comma_sep)


    # Read the text file using pandas
    data_frame = read_txt_file(txt_dot_decimal_comma_sep)
    # print(data_frame)
    print(data_frame.head())
    print(data_frame["Time"][0])
    # print(data_frame.columns)
    print("Data from the text file:")


# for daily in range(len(days)):
#
#     file_path = 'Data/Raw/' + days[daily] + ".xlsx"  # Update with the correct file path
#
#     dfd = pd.read_excel(file_path)
#
# print(dfd)
# print(dfd.columns)





