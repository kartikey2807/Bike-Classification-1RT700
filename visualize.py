import warnings
warnings.filewarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

overall_df = pd.read_csv('training_data_fall2024.csv')
## overall_df.info()

int_features = [
    'hour_of_day',
    'day_of_week',
    'month',
    'holiday',
    'weekday',
    'summertime',
    'snow'
]
con_features = [
    'temp',
    'dew',
    'humidity',
    'precip',
    'snowdepth',
    'windspeed',
    'cloudcover',
    'visibility'
]

bike_df, bike_validation_df = train_test_split(overall_df,test_size=0.2,
random_state=42,stratify=overall_df['increase_stock'])
## print(bike_df.shape)
## print(bike_df['increase_stock].value_counts())

## Exploration and Visualization
## 1. Draw pairplot for continueous features
## 2. Draw boxplot for label increaes_stock
## 3. Draw boxplot for categorical features
## 4. Figure out data imbalance
## 5. Find outliers

sns.set_style('darkgrid')

temp = con_features.copy()
temp.pop(-1) ## visibility
temp.pop(-1) ## cloudcover
temp.pop(-2) ## snowdepth

plt.figure(figsize=(15,15))
sns.pairplot(data=bike_df,vars=con_features,kind='scatter')
## ---------------------------------------------------------------------

plt.figure(figsize=(15,15))
plt.title('Heatmap estimate of features')
bike_cm = bike_df[:,con_features].corr()
sns.heatmap(bike_cm,annot=True,cmap='coolwarm',linewidth=0.2)
## ---------------------------------------------------------------------

## Corralation
## temp, dew, 0.87
## dew, humidity, 0.48
## humidity, cloudcover, 0.32
## humidity, windspeed, -0.34
## humidity, visibility, -0.38
## precip, visibility, -0.51
## ---------------------------------------------------------------------

fig,axes = plt.subplots(2,2,figsize=(15,15))
hue_label = [[None,'holiday'],
['weekday','summertime']]

for i in range(2):
    for j in range(2):
        axes[i,j].set_title('# Instances of bike demand')
        sns.countplot(data=bike_df,
        x=bike_df.loc[:,'increase_stock'],
        order=['low_bike_demand','high_bike_demand'],
        hue=hue_label[i][j],ax=axes[i,j])

        axes[i,j].set_xlabel('Bike demand')
        axes[i,j].set_ylabel('Count')
## ---------------------------------------------------------------------

fig,axes = plt.subplots(7,1,figsize=(15,40))

label_rep = {
    'low_bike_demand':0, 
    'high_bike_demand': 1}

for val in bike_df['day_of_week'].sort_values().unique():
    
    indexes = bike_df.index[bike_df.loc[:,'day_of_week'] == val]
    sns.lineplot(data=bike_df,
    x=bike_df.loc[:,'hour_of_day'],
    y=bike_df.loc[indexes,'increase_stock'].replace(label_rep),
    estimator='mean', ax=axes[int(val)], errorbar=('ci',False))
## ---------------------------------------------------------------------

plt.figure(figsize=(15,15))

sns.lineplot(data=bike_df, 
x=bike_df.loc[:,'hour_of_day'],
y=bike_df.loc[:,'increase_stock'].replace(label_rep),
hue='holiday', estimator='mean', 
errorbar=('ci',False))
## ---------------------------------------------------------------------

plt.figure(figsize=(15,5))

sns.lineplot(data=bike_df,
x=bike_df.loc[:,'month'],
y=bike_df.loc[:,'increase_stock'].replace(label_rep),
estimator='mean',errorbar=('ci',False))

plt.title('Instances of high bike demand vs Months')
plt.xlabel('Months')
plt.ylabel('High bike demand instances')
## ---------------------------------------------------------------------

con_features_np = np.array(con_features).reshape(4,2)
fig, axes = plt.subplots(4,2,figsize=(15,30))

for i in range(4):
    for j in range(2):
        axes[i,j].set_title(f'{con_features_np[i,j]} vs Bike demand')
        sns.boxplot(data=bike_df,
        x=bike_df.loc[:,'increase_stock'],
        y=bike_df.loc[:,con_features_np[i,j]],
        order = ['low_bike_demand','high_bike_demand'], 
        ax=axes[i,j], color="0.8")

        axes[i,j].set_xlabel('Bike Demand')
        axes[i,j].set_ylabel(f'{con_features_np[i,j]}')
## ---------------------------------------------------------------------

## labels vs continuous variables
## we can drop
con_features_np = np.array(con_features).reshape(4,2)
fig, axes = plt.subplots(4,2,figsize=(15,30))

for i in range(4):
    for j in range(2):
        axes[i,j].set_title(f'{con_features_np[i,j]} vs Bike demand')
        sns.violinplot(data=bike_df,
        x=bike_df.loc[:,'increase_stock'],
        y=bike_df.loc[:,con_features_np[i,j]],
        order = ['low_bike_demand','high_bike_demand'],
        ax=axes[i,j], linewidth=2,color="0.8")

        axes[i,j].set_xlabel('Bike Demand')
        axes[i,j].set_ylabel(f'{con_features_np[i,j]}')
## ---------------------------------------------------------------------

## Observation 1.1
## 1. temp and dew have linear +ve correlation.
## 2. Correlation threshold assumed at 0.8.
## 3. low_bike_demand (1050) >> high_bike_demand (230)
## 4. More high_bike_demand instances in summers than in winters.

## Observation 1.2
## Monday - Friday
## 1. high_bike_demand instances between 6AM and 9PM.
## 2. Graph flatlines <6AM and >9PM.
## 3. Count of instances peaks between 5PM and 7PM.
## Weekends
## 1. There are instances between 8AM and 8PM.
## Holidays
## 1. Instances between 9AM and 5PM.
## 2. Peak shifts between 10AM and 3PM.
## 3. Valley: could be the time when people are hanging out in pubs/bars

## Observation 1.3
## Conditions favoring high_bike_demand

## 1. high temperature
## 2. high dew point
## 3. low relative humidity
## 4. higher median windspeed
## 5. Outliers exists for precip, windspeed, snowdepth and visibility.
## ---------------------------------------------------------------------