import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
## ------------------------
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

def ce(df, T): ## cyclic encoding
    if T == 12:
        df = df - 1
    return np.sin(2*np.pi*df/T),np.cos(2*np.pi*df/T)

def density_plot(df, con_features): ## plot the density of continuous features
    fig, axes = plt.subplots(8,1, figsize = (15,50))

    for i in range(len(con_features)):
        sns.histplot(data=df,
        x=df.loc[:, con_features[i]], stat='density',
        color='blue', bins=50, ax=axes[i], kde=True)
    plt.show()