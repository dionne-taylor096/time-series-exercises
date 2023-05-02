#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import env
from env import host, user, password
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import wrangle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
# In[ ]:

from sklearn.model_selection import train_test_split

def train_val_test_split(df, features, target_regression, target_classification, test_size=0.3, random_state=42):
    # Separate the features and target variables
    X = df[features]
    y_regression = df[target_regression]
    y_classification = df[target_classification]

    # Split the data into training, validation, and testing sets for regression
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=test_size, random_state=random_state)
    X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(X_train_reg, y_train_reg, test_size=0.2, random_state=random_state)

    # Split the data into training, validation, and testing sets for classification
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_classification, test_size=test_size, random_state=random_state)
    X_train_cls, X_val_cls, y_train_cls, y_val_cls = train_test_split(X_train_cls, y_train_cls, test_size=0.2, random_state=random_state)
    
    return X_train_reg, X_val_reg, X_test_reg, y_train_reg, y_val_reg, y_test_reg, X_train_cls, X_val_cls, X_test_cls, y_train_cls, y_val_cls, y_test_cls

def add_productivity_class(df):
    def productivity_class(value):
        if value < 0.33:
            return 0  # low productivity
        elif value < 0.66:
            return 1  # medium productivity
        else:
            return 2  # high productivity

    df['productivity_class'] = df['actual_productivity'].apply(productivity_class)
    return df

def model_eval(df, target_column):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    feature_columns = [col for col in df.columns if col != target_column]

    # Split the data into train (70%), validation (15%), and test (15%) sets
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    validate, test = train_test_split(temp, test_size=0.5, random_state=42)

    # Separate the features and target for each dataset
    X_train, y_train = train[feature_columns], train[target_column]
    X_validate, y_validate = validate[feature_columns], validate[target_column]
    X_test, y_test = test[feature_columns], test[target_column]

    models = [
        ('Decision Tree', DecisionTreeClassifier(random_state=42)),
        ('Random Forest', RandomForestClassifier(random_state=42))
    ]

    best_model = None
    best_score = -np.inf

    for name, model in models:
        model.fit(X_train, y_train)
        score = model.score(X_validate, y_validate)
        if score > best_score:
            best_score = score
            best_model = (name, model)

    print(f"Best Model: {best_model[0]}, Accuracy score: {best_score:.3f}")
    
def baseline_metrics(df):
    # Split the data into features and target
    X = df.drop('actual_productivity', axis=1)
    y = df['actual_productivity']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set and evaluate performance
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    accuracy = accuracy_score(y_test.round(), y_pred.round())

    print("Baseline Metrics")
    print("R2 score:", r2)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("Accuracy score:", accuracy)
   
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

def model_all(df):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop('productivity_class', axis=1), df['productivity_class'], test_size=0.3, random_state=42)

    # Split the train set further into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    dt_model = DecisionTreeClassifier(random_state=42)
    rf_model = RandomForestClassifier(random_state=42)

    models = [dt_model, rf_model]

    for model in models:
        model.fit(X_train, y_train)

        # Predict on validation and test sets and evaluate performance
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        r2_val = r2_score(y_val, y_pred_val)
        mse_val = mean_squared_error(y_val, y_pred_val)
        rmse_val = np.sqrt(mse_val)
        accuracy_val = accuracy_score(y_val, y_pred_val.round())

        r2_test = r2_score(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        accuracy_test = accuracy_score(y_test, y_pred_test.round())

        # Predict on training set and evaluate performance
        y_pred_train = model.predict(X_train)

        r2_train = r2_score(y_train, y_pred_train)
        mse_train = mean_squared_error(y_train, y_pred_train)
        rmse_train = np.sqrt(mse_train)
        accuracy_train = accuracy_score(y_train, y_pred_train.round())

        print(f"\n{type(model).__name__} - Training Set:")
        print("R2 score:", r2_train)
        print("MSE:", mse_train)
        print("RMSE:", rmse_train)
        print("Accuracy score:", accuracy_train)

        print(f"\n{type(model).__name__} - Validation Set:")
        print("R2 score:", r2_val)
        print("MSE:", mse_val)
        print("RMSE:", rmse_val)
        print("Accuracy score:", accuracy_val)

        print(f"\n{type(model).__name__} - Test Set:")
        print("R2 score:", r2_test)
        print("MSE:", mse_test)
        print("RMSE:", rmse_test)
        print("Accuracy score:", accuracy_test)

