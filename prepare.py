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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
# In[ ]:

import pandas as pd


def clean_data(df):
    # Fill missing values in 'wip' with 0
    df['wip'] = df['wip'].fillna(0)

    # Find and replace "sweing" with "sewing"
    df['department'] = df['department'].replace('sweing', 'sewing')
    
    # remove leading and trailing whitespace
    df['department'] = df['department'].str.strip()

    # drop unnecessary columns
    df.drop(['day', 'quarter', 'date', 'actual_production', 'prod_capacity', 'actual_efficiency', 'department','month'], axis=1, inplace=True)

    return df

def data_dict_garment():
    data_dictionary = pd.DataFrame({
        'Column Name': ['date', 'day', 'quarter', 'department', 'team', 'no_of_workers', 'no_of_style_change', 
                        'targeted_productivity', 'smv', 'wip', 'over_time', 'incentive', 'idle_time', 'idle_men', 
                        'actual_productivity'],
        'Description': ['Date in MM-DD-YYYY', 'Day of the Week', 'A portion of the month. A month was divided into four quarters',
                        'Associated department with the instance', 'Associated team number with the instance', 
                        'Number of workers in each team', 'Number of changes in the style of a particular product',
                        'Targeted productivity set by the Authority for each team for each day.', 'Standard Minute Value, it is the allocated time for a task',
                        'Work in progress. Includes the number of unfinished items for products', 'Represents the amount of overtime by each team in minutes',
                        'Represents the amount of financial incentive (in BDT) that enables or motivates a particular course of action.',
                        'The amount of time when the production was interrupted due to several reasons',
                        'The number of workers who were idle due to production interruption',
                        'The actual % of productivity that was delivered by the workers. It ranges from 0-1.']
    })
    return data_dictionary




def custom_visual():

    # Set the Seaborn style to "darkgrid" for better visual contrast and readability
    sns.set_style("darkgrid")

    # Set the Matplotlib style to "dark_background" for a visually appealing dark theme
    plt.style.use('dark_background')

    # Configure the default float format in Pandas to display two decimal places
    pd.options.display.float_format = '{:20,.2f}'.format

    # Set the maximum column width in Pandas to display the entire content without truncation
    pd.set_option('display.max_colwidth', None)

    # Set the display width in Pandas to match the terminal/console width
    pd.set_option('display.width', None)

    # Reset the column header justification in Pandas to its default (left-aligned)
    pd.reset_option("colheader_justify", 'left')
    
    
def create_data_dict_df():
    data_dict = {
        "Module": [
            "pandas",
            "numpy",
            "seaborn",
            "wrangle",
            "acquire",
            "sklearn.model_selection.TimeSeriesSplit",
            "sklearn.metrics.mean_squared_error",
            "matplotlib.pyplot",
            "statsmodels.tsa.holtwinters.Holt",
            "statsmodels.tsa.seasonal.seasonal_decompose",
            "sklearn.model_selection.train_test_split",
            "sklearn.linear_model.LinearRegression",
            "sklearn.ensemble.RandomForestRegressor",
        ],
        "Description": [
            "Pandas is a popular library for handling and analyzing data in a table format. It is often used for data manipulation tasks, like filtering, sorting, or aggregating data. The pd alias is used as a shorthand to refer to Pandas functions in the rest of the code.",
            "Numpy is another popular library for working with arrays and numerical data. It provides functions for mathematical operations, like addition, multiplication, or statistical calculations. The np alias is used as a shorthand.",
            "Seaborn is a data visualization library built on top of Matplotlib (another visualization library). It simplifies the process of creating complex and informative graphs, making them visually appealing. The sns alias is used as a shorthand.",
            "This imports a custom module named wrangle, which likely contains functions for data preprocessing, like cleaning or transforming the data.",
            "This imports a custom module named acquire, which probably contains functions to obtain data from external sources, like databases or APIs.",
            "Scikit-learn is a widely used machine learning library, and this line imports the TimeSeriesSplit function, which is used to split time series data into training and testing sets for model validation.",
            "This imports the mean_squared_error function from Scikit-learn, which is a metric used to evaluate the performance of regression models by calculating the average squared difference between the true and predicted values.",
            "Matplotlib is a widely used data visualization library in Python. This line imports its pyplot submodule and assigns it the alias plt for easier reference.",
            "Statsmodels is a library for statistical modeling in Python. This line imports the Holt function, which is an implementation of the Holt-Winters forecasting method for time series data.",
            "This line imports the seasonal_decompose function from Statsmodels, which is used to analyze and visualize the trend, seasonality, and residual components of a time series.",
            "This imports the train_test_split function from Scikit-learn, which is used to randomly split a dataset into training and testing sets.",
            "This line imports the LinearRegression and LogisticRegression classes from Scikit-learn, which are used to create and fit linear and logistic regression models, respectively.",
            "This imports the RandomForestRegressor class from Scikit-learn, which is an ensemble method used for regression tasks, combining multiple decision trees to create a more accurate model.",

        ],
    }

    # Convert the dictionary to a DataFrame
    data_dict_df = pd.DataFrame(data_dict)
    
    return data_dict_df

def productivity_class(value):
    if value < 0.33:
        return 0  # low productivity
    elif value < 0.66:
        return 1  # medium productivity
    else:
        return 2  # high productivity
