"""
# Objective
* Predict if # of bikes needs to be increased at certain hours or not
* Data Exploration and Feaure Engineering
* Binary Classification Problem
* Train-validation set has *1600* records"""

import warnings
warnings.filterwarnings('ignore')

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

overall_df = pd.read_csv('/content/training_data_fall2024.csv')
## Specify your filepath

overall_df.info()

int_features = [
'hour_of_day',
'day_of_week',
'month',
'holiday',
'weekday',
'summertime',
'snow']

con_features = [
'temp',
'dew',
'humidity',
'precip',
'snowdepth',
'windspeed',
'cloudcover',
'visibility']

""" ## Data Split
Split data into:-
* Train set - 80%
* Validation set - 20%"""

from sklearn.model_selection import train_test_split

bike_df, bike_validation_df = train_test_split(overall_df,test_size=0.2, random_state=42,stratify=overall_df['increase_stock'])

print(bike_df.shape) ## check shape
print(bike_df['increase_stock'].value_counts()) ## proportional split?

print(bike_validation_df.shape) ## check shape
print(bike_validation_df['increase_stock'].value_counts())

""" ## Data Exploration and Visualization
* Draw *pairplot* for continuous features
* Draw *boxplot* for label *increase_stock*
* Draw *boxplot* for categorical features
* Figure out data imbalance and how to handle it
* Outliers
"""

sns.set_style('darkgrid')

temp = con_features.copy()
temp.pop(-1) # visibility
temp.pop(-1) # cloudcover
temp.pop(-2) # snowdepth


plt.figure(figsize=(15,15))
sns.pairplot(data=bike_df, vars=con_features, kind='scatter')

## verify using heatmap
## draw a covariance matrix
plt.figure(figsize=(15,15))

plt.title('Heatmap Estimate of features')
bike_cm = bike_df.loc[:,con_features].corr()
sns.heatmap(bike_cm, annot=True, cmap='coolwarm', linewidth=0.2)

## Look at distribution for the label
## And it's relation with continuous
## features. Plot it's relation with
## respect to temporal data.

fig, axes = plt.subplots(2, 2, figsize=(15,15))

hue_label = [
    [None, 'holiday'],
    ['weekday', 'summertime']]

for i in range(2):
    for j in range(2):
        axes[i,j].set_title('# Instances for bike demand')
        sns.countplot(data = bike_df, x = bike_df.loc[:, 'increase_stock'], order = ['low_bike_demand', 'high_bike_demand'], 
                      hue=hue_label[i][j],ax = axes[i,j])        
        axes[i,j].set_xlabel('Bike Demand')
        axes[i,j].set_ylabel('Count')

fig, axes = plt.subplots(7,1,figsize=(15,40)) # row-wise
label_rep = {'low_bike_demand':0, 'high_bike_demand': 1}

for val in bike_df['day_of_week'].sort_values().unique():
    axes[int(val)].set_title(f'Instances of high bike demand vs Hour [wrt. Day {val}]')
    indexes = bike_df.index[bike_df.loc[:,'day_of_week']==val]
    sns.lineplot(data = bike_df, x=bike_df.loc[:,'hour_of_day'], y=bike_df.loc[indexes,'increase_stock'].replace(label_rep),
                 estimator='mean', ax=axes[int(val)],errorbar=('ci',False))
    axes[int(val)].set_xlabel('Hour')
    axes[int(val)].set_ylabel('High bike demand instances')

plt.figure(figsize=(15,6))

plt.title('Instances of high bike demand vs Hour [wrt. Holiday]')
sns.lineplot(data=bike_df,x=bike_df.loc[:,'hour_of_day'],y=bike_df.loc[:,'increase_stock'].replace(label_rep),hue='holiday',
             estimator='mean',errorbar=('ci',False))

plt.xlabel('Hour')
plt.ylabel('High bike demand instances')

plt.figure(figsize=(15,5))

sns.lineplot(data=bike_df, x=bike_df.loc[:,'month'], y=bike_df.loc[:,'increase_stock'].replace(label_rep), estimator='mean',
             errorbar=('ci',False))

plt.title('Instances of high bike demand vs Months')
plt.xlabel('Months')
plt.ylabel('High bike demand instances')

## labels vs continuous variables
## we can drop
con_features_np = np.array(con_features).reshape(4,2)

fig, axes = plt.subplots(4,2,figsize=(15,30))

for i in range(4):
    for j in range(2):
        axes[i,j].set_title(f'{con_features_np[i,j]} vs Bike demand')
        sns.boxplot(data =bike_df, x=bike_df.loc[:,'increase_stock'], y=bike_df.loc[:,con_features_np[i,j]], 
                    order=['low_bike_demand','high_bike_demand'], ax=axes[i,j], color="0.8")

        axes[i,j].set_xlabel('Bike Demand')
        axes[i,j].set_ylabel(f'{con_features_np[i,j]}')

## labels vs continuous variables
## we can drop
con_features_np = np.array(con_features).reshape(4,2)

fig, axes = plt.subplots(4,2,figsize=(15,30))

for i in range(4):
    for j in range(2):
        axes[i,j].set_title(f'{con_features_np[i,j]} vs Bike demand')
        sns.violinplot(data=bike_df,x=bike_df.loc[:,'increase_stock'],y=bike_df.loc[:,con_features_np[i,j]],
                       order=['low_bike_demand','high_bike_demand'],ax=axes[i,j], linewidth=2, color="0.8")

        axes[i,j].set_xlabel('Bike Demand')
        axes[i,j].set_ylabel(f'{con_features_np[i,j]}')

"""
## Observation 1.1
* *temp* and *dew* have linear +ve correlation.
* Correlation threshold assumed at 0.8.
* low_bike_demand (1050) >> high_bike_demand (230)
* More *high_bike_demand* instances in summers than in winters.

## Observation 1.2

**Monday-Friday**

* *high_bike_demand* instances between 6AM and 9PM.
* Graph flatlines <6AM and >9PM.
* Count of instances peaks between 5PM and 7PM.

**Weekends**
* There are instances between 8AM and 8PM.

**Holidays**
* Instances between 9AM and 5PM.
* Peak shifts between 10AM and 3PM.
* **Valley** -> could be the time when people are hanging out in pubs and bars.

## Observation 1.3

**Conditions favoring high_bike_demand**
* high temperature
* high dew point
* low relative humidity
* higher median windspeed
* Outliers exists for **precip**, **windspeed**, **snowdepth** and **visibility**. \[Boxplot\]
---
"""