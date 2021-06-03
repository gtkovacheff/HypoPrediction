import pandas as pd
import numpy as np
import datetime
from datetime import timedelta


def extract_time_plus_minus(data):
    timedelta_zip = pd.Series([timedelta(hours=h,
                                     minutes=m,
                                     seconds=s) for h,m,s in zip(pd.to_datetime(data['Time']).dt.hour,
                 pd.to_datetime(data['Time']).dt.minute,
                 pd.to_datetime(data['Time']).dt.second)
                           ])

    data['Timestamp'] = pd.to_datetime(data['Date'])
    # data['Date'] = data['Timestamp'].dt.date
    data['TimeStart'] = data['Timestamp'].dt.time
    data['TimeFinish'] = (data['Timestamp'] + timedelta_zip).dt.time
    data['Date'] = data['Timestamp'].dt.date

    return data[['Timestamp', 'Date', 'Time', 'TimeStart', 'TimeFinish', 'Activity Type', 'Distance', 'Calories', 'Avg HR']]


def clean_data(data):
    #replace , with '' and -- with NaN
    data['Distance'] = [x.replace(',', '') for x in data['Distance']]
    data['Distance'] = [x.replace('--', 'NaN') for x in data['Distance']]

    data['Calories'] = [x.replace('--', 'NaN') for x in data['Calories']]
    data['Calories'] = [x.replace(',', '') for x in data['Calories']]

#transform the following columns into numeric format format
    for x in ['Distance', 'Calories', 'Avg HR']:
        data[x] = pd.to_numeric(data[x], 'coerce')

    #Distance above 200 should be divided by 1000 to be transformed into km
    data.loc[data['Distance'] > 200, 'Distance'] = data.loc[data['Distance'] > 200, 'Distance']/1000

    #remove NaN distances and above 45km clear outlier
    #remove NaN distances and above 45km clear oactivity = activity.dropna(subset=['Distance'])
    data = data[data['Distance'] < 49]

    # #fix activity on 2020-12-03 --> set time to be 1 hour and distance = 4.78 based on the website data
    # data.loc[data['Distance']==13.39, 'Time'] = "01:00:00"
    # data.loc[data['Distance']==13.39, 'Distance'] = 4.78
    # data.loc[data['Distance']==4.78]

    #remove 0 Calories, it is wrong data
    data = data[~data['Calories'].isna()]

    #impute Avg HR to be mean of Avg HR --> ## TODO: avg HR should be part of the pipeline
    data.loc[data['Avg HR'].isna(), 'Avg HR'] = round(np.mean(data['Avg HR']), 2)

    data.reset_index(inplace=True, drop=True)
    return data


def load_clean_bg(path_to_bg_data="Data/BloodG.txt", fix_time=True, safe=False):
    bg = pd.read_csv(path_to_bg_data, skiprows=3, sep='\t')
    bg = bg.loc[:, ['Time', 'Glucose (mmol/L)']].\
        rename(columns={'Time': 'Date',
                        'Glucose (mmol/L)': 'Glucose'})

    if fix_time:
    #fix the date substracting the timedelta from last day in the dataset
        bg.loc[:, 'Date'] = pd.to_datetime(bg['Date'])
        bg['TimeBG'] = bg['Date'].dt.time
        bg.loc[:, 'DateBG'] = bg.loc[:, 'Date'].dt.date
        delta_date = datetime.datetime(2021, 4, 23).date()
        # delta_date = datetime.datetime.now().date()
        delta = bg.loc[1047, 'DateBG'] - delta_date
        bg.loc[:, 'DateBG'] = list(map(lambda x: x - timedelta(delta.days), bg['DateBG']))

    #rearrange the columns
    bg = bg[['DateBG', 'TimeBG', 'Glucose']]

    ##safe the data
    if safe:
        bg.to_csv('Data/Prepped/BloodGlucose' + str(datetime.datetime.now().date()) + '.csv')

    return bg

def to_timedelta(x):
    h = pd.to_datetime(x).hour
    m = pd.to_datetime(x).minute
    s = pd.to_datetime(x).second
    return timedelta(hours=h, minutes=m, seconds=s)
