import pandas as pd
import numpy as np
import datetime
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import GradientBoostingRegressor


from func import extract_time_plus_minus, clean_data, load_clean_bg, to_timedelta, one_row_per_date, transform_glucose

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

activity = clean_data(activity)
activity = extract_time_plus_minus(activity)
activity = activity.loc[activity['TimeStart'] > datetime.time(12)]


for t in activity['Date']:
    print(f"{t} has shape {activity.loc[activity['Date'] == t].shape}")



#group the activity data into dates and apply the one_row_per_date function to every group :)
activity_grouped = activity.groupby('Date')
empty_df = pd.DataFrame()
for name, group in activity_grouped:
    print(name, ' has min data -- >>', group['TimeStart'].min())
    empty_df = pd.concat([empty_df, one_row_per_date(group)], axis=0)

activity_prepped = empty_df.reset_index(drop=True)

# activity_prepped.to_csv('Data/Prepped/Activity_prepped' + str(datetime.datetime.now().date()) + '.csv')

#merge the two data sets
data = pd.merge(activity_prepped, bg, how='inner', left_on='Date', right_on='DateBG')
data = data.drop(['DateBG'], axis=1).rename(columns={'Time': 'Duration'})#, drop DateBH and rename Time -> Duration
data['TimeBG'] = [datetime.time(hour=int(h), minute=int(m)) for h, m in
                  zip([x.split(':')[0] for x in data['TimeBG']], [x.split(':')[1] for x in data['TimeBG']])]
# #explore the data structure for each day
# for i in data['Date'].unique():
#     print(f'{i} date has {data.loc[data["Date"] == i].shape}')

# the goal is to have Date, Duration, TimeStart, TimeFinish, Activity Type, Distance, Calories, Avg HR, Glucose Prior, Glucose After
#lets use the sample 20210422

data_grouped = data.groupby('Date')

empty_df = pd.DataFrame()
for name, group in data_grouped:
    print(name)
    empty_df = pd.concat([empty_df, transform_glucose(group)], axis=0)

data = empty_df.astype({'Distance': 'double',
                        'Calories': 'double',
                        'Avg HR': 'double',
                        'PreGlucose': 'double',
                        'PreGlucose': 'double',
                        'PostGlucose': 'double'}).\
    reset_index(drop=True).\
    drop('PostGlucoseTime', axis=1)


data

# numeric_features = ['Distance', 'Calories', 'Avg HR', 'PreGlucose', 'PostGlucose']
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
standart_scaler_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))])

data['PostGlucose'] = numeric_transformer.fit_transform(data['PostGlucose'].values.reshape(-1, 1)).reshape(-1)

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, selector(dtype_include='float64')),
    ('num_scale', standart_scaler_transformer, selector(dtype_include='float64')),
    ('cat', categorical_transformer, selector(dtype_exclude='object'))
])

reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', GradientBoostingRegressor())])

X_train, y_train = data[['Activity Type', 'Distance', 'Calories', 'Avg HR', 'PreGlucose']], data[['PostGlucose']]

reg_pipeline.fit(X_train, y_train)
reg_pipeline.score(X_train, y_train)

reg_pipeline.predict(X_train.loc[0].to_frame().transpose())
y_train.loc[0]

#test 1 is ok
test = {'Activity Type': 'Cardio', 'Distance':0, 'Calories': 74 , 'Avg HR': 102, 'PreGlucose':9.2}
reg_pipeline.predict(pd.DataFrame(test, index=[0]))

#predicts:4.96 True: 5.7

#test 2 is ok
test = {'Activity Type': 'Running', 'Distance':3.74, 'Calories': 618 , 'Avg HR': 150, 'PreGlucose':12.2}
reg_pipeline.predict(pd.DataFrame(test, index=[0]))

#predicts:6.4 True: 7.2

#test 3 doesn't work as Distance --> np.NaN is invalid
test = {'Activity Type': 'Running', 'Distance':np.NAN, 'Calories': 618 , 'Avg HR': 150, 'PreGlucose':12.2}
reg_pipeline.predict(pd.DataFrame(test, index=[0]))

# safe the model
with open('saved_models/gbm_1.pkl', 'wb') as m:
    pickle.dump(reg_pipeline, m)

#TO BE CONTINUED
with open('saved_models/gbm_1.pkl', 'rb') as m:
    reg_pipeline_loaded = pickle.load(m)

# Graphics on Blood Glucose
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