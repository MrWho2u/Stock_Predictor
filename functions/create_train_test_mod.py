# General packages
import pandas as pd
import numpy as np
import datetime as dt
np.random.seed(42)

from .vix_mod import vix_analysis
from .spy_mod import spy_analysis
from .econ_mod import get_econ_data
from .sent_mod import market_sent

def create_train_test_tables ():
    spy_df = spy_analysis()
    econ_df = get_econ_data()
    vix_df = vix_analysis()
    sentiment_df = market_sent()

    # Set Up DataFrame for Testing
    X_prep = pd.concat([vix_df, spy_df], axis=1)
    X_prep['y']=X_prep['spy_change'].shift(-1)*100
    X_prep = X_prep.dropna()
    X_prep = pd.concat([X_prep, econ_df, sentiment_df], axis=1)
    X_prep = X_prep.dropna(subset='spy_close')
    X_prep[np.isnan(X_prep)] = 0
    X_prep = X_prep.drop(columns=['spy_close'])

    X_full = X_prep.drop(columns=['y'])
    y_full = X_prep[['y']]
    
    #define train period
    start_train = X_full.index.min()
    end_train = dt.datetime.strptime('2020-01-01', '%Y-%m-%d').date()

    # Define test period
    start_test = end_train
    end_test = X_full.index.max()

    #Create train Data Frames
    X_train = X_full.loc[start_train: end_train]
    y_train = y_full.loc[start_train: end_train]
    X_prep_train = X_prep.loc[start_train: end_train]

    # Create test DataFrames
    X_test = X_full.loc[start_test: end_test]
    y_test = y_full.loc[start_test: end_test]
    X_prep_test = X_prep.loc[start_test: end_test]
    
    return X_train, y_train, X_test, y_test, X_prep_train, X_prep_test