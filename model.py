import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

overall_df = pd.read_csv('training_data_fall2024.csv')

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
## ---------------------------------------------------------------------

from sklearn.model_selection import train_test_split
bike_df, bike_validation_df = train_test_split(overall_df,test_size=0.2,
random_state=42,stratify=overall_df['increase_stock'])

## Data Transformation
## Remove Nan values
## 1. Fix Labels
## 2. Convert hour, day and month into cyclic encoding
## 3. Convert the remaining categories into One-Hot
## 4. Scale the continuous features
## 5. Drop snow and dew columns

label_rep = {
    'low_bike_demand':0, 
    'high_bike_demand': 1}

bike_df['increase_stock'] = \
bike_df['increase_stock'].replace(label_rep)

bike_validation_df['increase_stock'] = \
bike_validation_df['increase_stock'].replace(label_rep)
## ---------------------------------------------------------------------

## Cyclic Encoding
## 1. convert temporal data into cyclic form
## 2. decompose into sine and cosine parts
## 3. applied to hour, day and month

## One-Hot Encoding
## 1. applied to holiday, weekday and summertime

def cyclical_encoding(df, T):
    return np.sin(2*np.pi*df/T), np.cos(2*np.pi*df/T)

bike_df['hour_of_day_sin'], bike_df['hour_of_day_cos'] = \
cyclical_encoding(bike_df.loc[:,'hour_of_day'], T=24)
bike_validation_df['hour_of_day_sin'], \
bike_validation_df['hour_of_day_cos'] = \
cyclical_encoding(bike_validation_df.loc[:,'hour_of_day'], T=24)

bike_df['day_of_week_sin'], bike_df['day_of_week_cos'] = \
cyclical_encoding(bike_df.loc[:,'day_of_week'], T=7)
bike_validation_df['day_of_week_sin'], \
bike_validation_df['day_of_week_cos'] = \
cyclical_encoding(bike_validation_df.loc[:,'day_of_week'], T=7)

bike_df['month_sin'],bike_df['month_cos'] = \
cyclical_encoding(bike_df.loc[:, 'month'] - 1, T=12)
bike_validation_df['month_sin'],bike_validation_df['month_cos'] = \
cyclical_encoding(bike_validation_df.loc[:, 'month'] - 1, T=12)
## ---------------------------------------------------------------------

## delete the original columns
bike_df = bike_df.drop(['hour_of_day', 'day_of_week', 'month'], axis=1)
bike_validation_df = bike_validation_df.drop(['hour_of_day', 
'day_of_week', 'month'], axis=1)

bike_df = pd.get_dummies(bike_df, 
columns=['holiday','weekday','summertime'], dtype=int, drop_first=True)
bike_validation_df = pd.get_dummies(bike_validation_df, 
columns=['holiday','weekday','summertime'], dtype=int, drop_first=True)
## ---------------------------------------------------------------------

def density_plot(df):
    fig, axes = plt.subplots(8,1,figsize=(15,50))

    for i in range(len(con_features)):
        axes[i].set_title(f'Probability Density[{con_features[i]}]')
        sns.histplot(data=bike_df, x=bike_df.loc[:,con_features[i]], 
        stat='density', color='blue', bins=50, ax=axes[i], kde=True)
    plt.show()
## ---------------------------------------------------------------------

density_plot(bike_df)
density_plot(bike_validation_df)

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
bike_df.loc[:,con_features] = scale.fit_transform(X=bike_df.loc[:,con_features])
bike_validation_df.loc[:,con_features] = scale.transform(X=bike_validation_df.loc[:,con_features])
## ---------------------------------------------------------------------

density_plot(bike_df)
density_plot(bike_validation_df)
## ---------------------------------------------------------------------

bike_df = bike_df.drop(['snow','dew'], axis=1)
bike_validation_df = bike_validation_df.drop(['snow','dew'], axis=1)

bike_df = bike_df.drop(['weekday_1','summertime_1'], axis=1)
bike_validation_df = bike_validation_df.drop(['weekday_1','summertime_1'], axis=1)

## Data Modeling
## 1. Apply Naive classfier
## 2. Will always predict 'low_bike_demand'
## 3. Something that forms a benchmark

## Metrics
## 1. Accuracy
## 2. F1 score
## 3. Precision and Recall
## ---------------------------------------------------------------------

X_train = np.array(bike_df.drop(['increase_stock'], axis=1))
y_train = np.array(bike_df['increase_stock'])

X_valid = np.array(bike_validation_df.drop(['increase_stock'], axis=1))
y_valid = np.array(bike_validation_df['increase_stock'])


from sklearn.dummy import DummyClassifier



dummy_classifier = DummyClassifier(strategy='constant', constant=0, random_state=42)
dummy_classifier.fit(X_train, y_train)
## ---------------------------------------------------------------------

## Vanilla regression
## Hyper-parameters to fine-tune
## penalty
## lambda
## solver
## max_iter
## class_weights

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, make_scorer, f1_score



params = {
'penalty': ['l1','l2', 'elasticnet', None],
'C': [0.02,0.05,0.1,0.2,0.5,1,5,10],
'max_iter': [1,3,5,10,50,75,100],
'class_weight': [None,'balanced',{1:1.25,0:1},{1:1.5,0:1},{1:2,0:1},{1:2.25,0:1},{1:2.5,0:1}],
'solver': ['lbfgs','liblinear','newton-cg','newton-cholesky','saga']}
## ---------------------------------------------------------------------

logistic_regression = LogisticRegression(random_state=42)
grid_search = GridSearchCV(estimator=logistic_regression, 
param_grid=params,
cv=5, 
verbose=1, 
scoring='f1')

grid_search.fit(X_train, y_train)
## ---------------------------------------------------------------------

## SMOTE
## oversampling the high_bike_demand class

from imblearn.over_sampling import SMOTENC
over_sample = SMOTENC(categorical_features= [13], sampling_strategy=0.4,
random_state=42)
X_train_resample, y_train_resample = over_sample.fit_resample(X_train, y_train)

grid_search.fit(X_train_resample, y_train_resample)
## ---------------------------------------------------------------------

## Feature expansion
## Numerical features to combine:-
## temp and humidity
## temp and precip
## temp and windspeed
## temp and snowdepth
## humidity and precip
## humidity and windspeed
## precip and snowdepth
## precip and cloudcover
## precip and windspeed
## precip and visibility
## windspeed and cloudcover
## ---------------------------------------------------------------------
## Categorical features to combine:-
## 
## hour_of_day and day_of_week
## day_of_week and month
## hour_of_day and month

def feature_polynomial(X):

    return np.concatenate((X,
        (X[:,0]*X[:,1]).reshape((X.shape[0],1)),
        (X[:,0]*X[:,2]).reshape((X.shape[0],1)),
        (X[:,0]*X[:,4]).reshape((X.shape[0],1)),
        (X[:,0]*X[:,3]).reshape((X.shape[0],1)),
        (X[:,1]*X[:,2]).reshape((X.shape[0],1)),
        (X[:,1]*X[:,4]).reshape((X.shape[0],1)),
        (X[:,2]*X[:,3]).reshape((X.shape[0],1)),
        (X[:,2]*X[:,5]).reshape((X.shape[0],1)),
        (X[:,2]*X[:,4]).reshape((X.shape[0],1)),
        (X[:,2]*X[:,6]).reshape((X.shape[0],1)),
        (X[:,4]*X[:,5]).reshape((X.shape[0],1)),
        (X[:,7]*X[:,9]).reshape((X.shape[0],1)),
        (X[:,8]*X[:,10]).reshape((X.shape[0],1)),
        (X[:,7]*X[:,11]).reshape((X.shape[0],1)),
        (X[:,8]*X[:,12]).reshape((X.shape[0],1)),
        (X[:,9]*X[:,11]).reshape((X.shape[0],1)),
        (X[:,10]*X[:,12]).reshape((X.shape[0],1))), axis=1)

X_train_fe = feature_polynomial(X_train)
X_valid_fe = feature_polynomial(X_valid)

grid_search.fit(X_train_fe, y_train)
## ---------------------------------------------------------------------