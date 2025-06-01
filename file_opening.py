#importing needed libraries

import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone



'''
===================================================================================
@desc - Creates dataframe of given USGS data csv
@param - 
    {String} data_type either 'ADVM', 'SSC_Q', or 'Qall'
    {S Array} ids list of strings of USGS site numbers
    {Boolean} [printo = False] Prints output of dataframes
@output -
    {data frame} df dataframe of csv file read

@potential future upgrades - 
        maybe a little less weird having to input _ADVM or _SSC_Q from interactive code
        make it so you pass in a dictionary for it to add into
===================================================================================
'''

def create_USGS_dfs(data_type, ids, printo = False):
    df = {} # empty dictionary to put dataframes into
    current_path = os.getcwd()
    for i in ids:
        df[i] = pd.read_csv(current_path + r'/USGS_data/'+i+'_'+data_type+'.csv')
    if printo == True:  
        for key in df.keys():
            print("\n" +"="*40)
            print(key)
            print("-"*40)
            print(df[key])

    return df

'''
===================================================================================
@desc - Creates array of array for SNR and Amp beams
@param - 
    {df} dataframe to read from
    {S Array} names start stop names for array of arrays (length = 2)
@output -
    {array of arrays} tada [m,x] matrix of m cells with x readings

@potential future upgrades - 
        figure out number of beams and cell (or have inputed and index accordingly)
        Put in whole data frame and have it output the number of arrays of arrays wants (e.g. 2 SNR 2 Amp each 10 cells)
        pass in dictionary of df and perform them all
       
===================================================================================
'''

def beam_array(df, names):
    a = df.columns.get_loc(names[0])
    b = df.columns.get_loc(names[1])
    tada = df.iloc[:,a:b+1].to_numpy().transpose()
    
    return tada

'''
===================================================================================
@desc - Converts read in site ids from number to string (and adds 0 to begining)
@param - 
    {Number} x value to change
@output -
    {String} USGS 8 digit site name
===================================================================================
'''
def toName(x):
    return '%08d' %x

'''
===================================================================================
@desc - Converts excel serial date time to YYYY-MM-DD HH:MM:SS format and tags as UTC
##### DOESN'T AUTOMATICALLY MAKE CONVERSION TO UTC FROM WHATEVER INPUTED TIMEZONE, JUST TAGS WHATEVER CALCULATED TIME AS UTC #####
@param - 
    {Array of floats} excel_date: Serial date time
    {int} tz_hour_offset hour timezone offset from UTC (can be + or -, e.g. EDT is -4) (default 0)
    {int} tz_minute_offset: minute timezone offset from UTC (default 0)
@output -
    {datetime.datetime} date time format YYYY-MM-DD HH:MM:SS
===================================================================================
'''
def serialTimeToDatetime(excel_date, tz_hour_offset = 0, tz_minute_offset = 0):
#consolidated from https://stackoverflow.com/questions/31359150/convert-date-from-excel-in-number-format-to-date-format-python
    dts = np.array([])
    for x in excel_date:

        dt = datetime.fromordinal(datetime(1900, 1, 1).toordinal() + int(x) - 2)
        hour, hourSecs = divmod(x % 1, 1)
        minute, second = divmod(hourSecs * 60, 1)
        dt = dt.replace(hour=int(hour) - tz_hour_offset, minute=int(minute) - tz_minute_offset, second=int(second * 60), tzinfo=timezone.utc)
        dts = np.append(dts, dt)
    return dts