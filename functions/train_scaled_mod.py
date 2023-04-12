# import libraries
import pandas as pd
import datetime as dt
from sklearn.preprocessing import StandardScaler
import numpy as np
np.random.seed(42)

def scale_train (X_train, X_prep_train):
    # fit Scale X Variables
    X_train.std(ddof=1)
    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_scaler = X_scaler.transform
    
    # Scale X Variables
    X_train_scaled = X_scaler(X_train) 
    
    # scale based on volatility
    X0_train_scaled = X_scaler(X_train[X_train['labels']==0])
    X1_train_scaled = X_scaler(X_train[X_train['labels']==1])
    X2_train_scaled = X_scaler(X_train[X_train['labels']==2])
    
    y_train = X_prep_train['y'].values
    y0_train = X_prep_train[X_prep_train['labels']==0]['y']
    y1_train = X_prep_train[X_prep_train['labels']==1]['y']
    y2_train = X_prep_train[X_prep_train['labels']==2]['y']
    
    return X_train_scaled, X0_train_scaled, X1_train_scaled, X2_train_scaled, y_train, y0_train, y1_train, y2_train