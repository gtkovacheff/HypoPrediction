import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
from pdb import set_trace

from func import extract_time_plus_minus, clean_data, load_clean_bg, to_timedelta

#some predefined options for pandas
with open('pandasOptions.txt', 'r') as f:
    for line in f:
        pd.set_option(line.split("=")[0], int(line.split("=")[1]))

#read the cvs
# bg = load_clean_bg(safe = True)

bg = pd.read_csv('Data/Prepped/BloodGlucose.csv', index_col=0)
bg['Date'] = pd.to_datetime(bg['Date']).dt.date
bg.rename(columns={'Time': 'TimeBG',
                   'Date': 'DateBG'}, inplace=True)

#read the activity data
activity = pd.read_csv('Data/Activities.csv')
activity.drop(columns=['Grit', 'Flow', 'Bottom Time', 'Min Temp', 'Surface Interval', 'Decompression', 'Max Temp',
                       "Avg Vertical Ratio", "Avg Vertical Oscillation", "Training Stress Score®", 'Title', 'Favorite',
                       'Avg Run Cadence', 'Max Run Cadence', 'Avg Pace', 'Best Pace', 'Elev Gain', 'Elev Loss', 'Avg Stride Length', 'Climb Time',
                       'Max HR', 'Best Lap Time', 'Number of Laps', 'Total Strokes', 'Avg. Swolf', 'Avg Stroke Rate', 'Total Reps', 'Total Sets'], inplace=True)


#TODO feature engineering
#TODO Time column in minutes, think about it

activity = clean_data(activity)
activity = extract_time_plus_minus(activity)

for t in activity['Date']:
    print(f"{t} has shape {activity.loc[activity['Date'] == t].shape}")

tst = activity[activity['Date'] == pd.to_datetime('2021-04-22')]

def one_row_per_date(data):
    # set_trace()
    data_new = data.copy()
    data_new.loc[:, 'TimeStart'] = min(data_new.loc[:, 'TimeStart'])
    data_new.loc[:, 'TimeFinish'] = max(data_new.loc[:, 'TimeFinish'])
    list_activity_type = data_new.sort_values(by=['Activity Type'], axis=0).loc[:, 'Activity Type'].tolist()
    data_new.loc[:, ['Activity Type']] = ['_'.join(list_activity_type)]
    data_new['Time'] = list(map(lambda x: to_timedelta(x), list(data_new['Time'])))

    data_new = data_new.groupby(['Date', 'TimeStart', 'TimeFinish', 'Activity Type']).\
        agg({'Time': 'sum',
             'Distance': 'sum',
             'Calories': 'sum',
             'Avg HR': 'mean'}).\
        reset_index()

    return data_new

#group the activity data into dates and apply the one_row_per_date function to every group :)
activity_grouped = activity.groupby('Date')
empty_df = pd.DataFrame()
for name, group in activity_grouped:
    print(name)
    empty_df = pd.concat([empty_df, one_row_per_date(group)], axis=0)

activity_prepped = empty_df.reset_index(drop=True)

# activity_prepped.to_csv('Data/Prepped/Activity_prepped' + str(datetime.datetime.now().date()) + '.csv')

#merge the two data sets
data = pd.merge(activity_prepped, bg, how='inner', left_on='Date', right_on='DateBG')
data = data.drop(['DateBG'], axis=1).rename(columns={'Time': 'Duration'})#, drop DateBH and rename Time -> Duration

#TODO add in the function from func.py
data['TimeBG'] = [datetime.time(hour=int(h), minute=int(m)) for h,m in zip([x.split(':')[0] for x in data['TimeBG']],
                                                  [x.split(':')[1] for x in data['TimeBG']])]

# #explore the data structure for each day
# for i in data['Date'].unique():
#     print(f'{i} date has {data.loc[data["Date"] == i].shape}')

# the goal is to have Date, Duration, TimeStart, TimeFinish, Activity Type, Distance, Calories, Avg HR, Glucose Prior, Glucose After
#lets use the sample 20210422
temp = data.loc[data['Date'] == pd.to_datetime('2021-04-22')]


#TODO PRE POST GLUCOSE COLUMNS
temp['TimeStart'][173] > temp['TimeBG'][173]










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