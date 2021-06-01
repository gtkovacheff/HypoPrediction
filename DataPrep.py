import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
#some predefined options for pandas
with open('pandasOptions.txt', 'r') as f:
    for line in f:
        pd.set_option(line.split("=")[0], int(line.split("=")[1]))

#read the cvs
# bg = pd.read_csv("Data/BloodG.txt", skiprows=3, sep='\t')
# bg.columns
#
# #extract only those with relevant info
# bg = bg[['Time', 'Glucose (mmol/L)']]
# bg = bg.rename(columns={'Time': 'Date', 'Glucose (mmol/L)': 'Glucose'})
#
# #fix the date substracting the timedelta from last day in the dataset
# bg.loc[:, 'Date'] = pd.to_datetime(bg['Date'])
# bg['Time'] = bg['Date'].dt.time
# bg.loc[:, 'Date'] = bg.loc[:, 'Date'].dt.date
#
# delta_date = datetime.datetime(2021, 4, 23).date()
# delta = bg.loc[1047, 'Date'] - delta_date
#
# #reassign the date
# bg.loc[:, 'Date'] = list(map(lambda x: x - datetime.timedelta(delta.days), bg['Date']))
#
#
# #rearrange the columns
# bg = bg[['Date', 'Time', 'Glucose']]
#
# #safe the data
# bg.to_csv('Data/Prepped/BloodGlucose.csv')
bg = pd.read_csv('Data/Prepped/BloodGlucose.csv', index_col=0)
bg['Date'] = pd.to_datetime(bg['Date']).dt.date

#read the activity data
activity = pd.read_csv('Data/Activities.csv')
activity.drop(columns=['Grit', 'Flow', 'Bottom Time', 'Min Temp', 'Surface Interval', 'Decompression', 'Max Temp',
                       "Avg Vertical Ratio", "Avg Vertical Oscillation", "Training Stress Score®", 'Title', 'Favorite',
                       'Avg Run Cadence', 'Max Run Cadence', 'Avg Pace', 'Best Pace', 'Elev Gain', 'Elev Loss', 'Avg Stride Length', 'Climb Time',
                       'Max HR', 'Best Lap Time', 'Number of Laps', 'Total Strokes', 'Avg. Swolf', 'Avg Stroke Rate', 'Total Reps', 'Total Sets'], inplace=True)

#replace , with '' and -- with NaN
activity['Distance'] = [x.replace(',', '') for x in activity['Distance']]
activity['Distance'] = [x.replace('--', 'NaN') for x in activity['Distance']]

activity['Calories'] = [x.replace('--', 'NaN') for x in activity['Calories']]
activity['Calories'] = [x.replace(',', '') for x in activity['Calories']]

#transform the following columns into numeric format format
for x in ['Distance', 'Calories', 'Avg HR']:
    activity[x] = pd.to_numeric(activity[x], 'coerce')

#Distance above 200 should be divided by 1000 to be transformed into km
activity.loc[activity['Distance'] > 200, 'Distance'] = activity.loc[activity['Distance'] > 200, 'Distance']/1000

# activity['Distance'].apply(lambda x: x + 10)
# activity['Distance'].transform(lambda x: x + 10)


#remove NaN distances and above 45km clear outlier
#remove NaN distances and above 45km clear oactivity = activity.dropna(subset=['Distance'])
activity = activity[activity['Distance'] < 49]

#remove 0 Calories, it is wrong data
activity = activity[~activity['Calories'].isna()]

#impute Avg HR to be mean of Avg HR
activity.loc[activity['Avg HR'].isna(), 'Avg HR'] = round(np.mean(activity['Avg HR']), 2)


# Feature engineering
activity['TimePlus4'] = (pd.to_datetime(activity['Date']) + timedelta(hours=4)).dt.time
activity['TimeMinus3'] = (pd.to_datetime(activity['Date']) - timedelta(hours=3)).dt.time

activity['Time'] = pd.to_datetime(activity['Date']).dt.time
activity['Timestamp'] = pd.to_datetime(activity['Date'])
activity['Date'] = pd.to_datetime(activity['Date']).dt.date
bg['Date']
columns_reorder = ["Activity Type", "Timestamp", "Date", "Time", 'TimePlus4', 'TimeMinus3', "Distance", "Calories", "Avg HR"]
activity = activity[columns_reorder]

#merge the two data sets
data = pd.merge(activity, bg, how='inner', on=['Date'])

for i in data['Date'].unique():
    print(f'{i} date has {data.loc[data["Date"] == i].shape}')


temp = data.loc[data['Date'] == pd.to_datetime('2021-04-22')]

def transform_data(df):
    df['Bool_1'] = (df['TimeMinus3'] < df['Time_y'])
    df['Bool_2'] = (df['Time_y'] < df['TimePlus4'])
    df['Bool'] = (df['Bool_1'] & df['Bool_2'])

    df = df[df['Bool'] == True]

    df = df.drop(['Bool_1', 'Bool_2', 'Bool'], axis=1)

    activity_type = "_".join(temp['Activity Type'].unique())
    prior_glucose = df.loc[temp['Time_y'] == min(df['Time_y']), 'Glucose'].unique()[0]
    post_glucose = df.loc[temp['Time_y'] == max(df['Time_y']), 'Glucose'].unique()[0]


    df = df.drop(["Activity Type", 'Glucose', 'Time_y'], axis=1).drop_duplicates().groupby(by=['Date']).\
        agg({'Distance': np.sum,
            'Calories': np.sum,
            'Avg HR': np.mean}).\
        reset_index()

    df['Activity'] = activity_type
    df['Post_Glucose'] = post_glucose
    df['Pre_Glucose'] = prior_glucose

    return df

df = pd.DataFrame()
df.append(data.iloc[1])
pd.concat([data.iloc[1], data.iloc[1]], axis=1)


for x in data['Date'].unique():
    print(x)
    temp = transform_data(x)
    df = df.append(temp)




# TO DO NEXT
#
# bg.loc[1, 'Time'] > datetime.time(4, 40)
#
# #define the time ranges:
# # Morning between 7:00 and 11:00
# # Lunch_1 between 11:00 and 15:00
# # Lunch_2 between 15:00 and 19:00
# # Dinner between 19:00 and 23:00
# # Night between 23:00 and 7:00
#
# def assign_time_category(x):
#     if x >= datetime.time(7) and x < datetime.time(11):
#         return 'Morning'
#     elif x >= datetime.time(11) and x < datetime.time(15):
#         return 'Early Lunch'
#     elif x >= datetime.time(15) and x < datetime.time(19):
#         return 'Late Lunch'
#     elif x >= datetime.time(19) and x < datetime.time(23):
#         return 'Dinner'
#     else:
#         return 'Night'
#
# # bg['category'] = list(map(apply(assign_time_category, bg['Time'])))
#
# bg['Category'] = bg.Time.apply(lambda x: assign_time_category(x))
#
#
# bg.groupby(['Category'])['Glucose'].apply(lambda x:calc_hypo(x))
#
#
# def calc_mean(x):
#     return np.mean(x)
#
#
# def calc_hypo(x, cnt=0):
#     if x <= 4:
#         cnt+=1
#     return(cnt)

import numpy as np
arr = np.array([[0, 1, 0, 0],
                [3, 2, 1, 0],
                [-2, 3, 2, 1],
                [1, -6, 0, 1]])

np.linalg.det(arr)

arr = np.array([[-3, 9],
               [6, -17]])

A_inv = np.linalg.inv(arr)
b = np.array([1, 5, 0])

np.dot(b, A_inv)