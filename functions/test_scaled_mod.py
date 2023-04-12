# import libraries
import pandas as pd
import datetime as dt
from sklearn.preprocessing import StandardScaler
import numpy as np
np.random.seed(42)

def scale_test (X_train, X_test, X_prep_test):
    # fit Scale X Variables
    X_train.std(ddof=1)
    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_scaler = X_scaler.transform
    
    # Scale X Variables
    X_test_scaled = X_scaler(X_test)
    
    # Sub variables
    X0_test_scaled = X_scaler(X_test[X_test['labels']==0])
    X1_test_scaled = X_scaler(X_test[X_test['labels']==1])
    X2_test_scaled = X_scaler(X_test[X_test['labels']==2])

    y_test = X_prep_test['y'].values
    y0_test = X_prep_test[X_prep_test['labels']==0]['y']
    y1_test = X_prep_test[X_prep_test['labels']==1]['y']
    y2_test = X_prep_test[X_prep_test['labels']==2]['y']

    return X_test_scaled, X0_test_scaled, X1_test_scaled, X2_test_scaled, y_test, y0_test, y1_test, y2_test