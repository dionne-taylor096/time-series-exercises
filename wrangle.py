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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
# In[ ]:


# In[ ]:

def convert_to_integer(df, columns):
    for col in columns:
        df[col] = df[col].fillna(-1).astype(int)
        df.loc[df[col] == -1, col] = np.nan
    return df

def standardize_text(df, columns):
    for col in columns:
        df[col] = df[col].str.upper().str.strip()
    return df

def detect_column_types(df):
    """
    Returns a dictionary with column names grouped by their data types.
    """
    column_info = df.dtypes.groupby(df.dtypes).groups
    column_groups = {}
    for dtype, column_list in column_info.items():
        column_groups[dtype] = column_list.tolist()
    return column_groups


def encode_categorical_columns(df, categorical_columns, encoding_method='ordinal'):
    """
    Encodes categorical columns using the specified encoding method.
    """
    if encoding_method == 'ordinal':
        encoder = OrdinalEncoder()
        df[categorical_columns] = encoder.fit_transform(df[categorical_columns])
    # Add other encoding methods if needed
    return df


def change_numerical_columns_datatype(df, numerical_columns, datatype='float64'):
    """
    Changes the datatype of numerical columns.
    """
    for column in numerical_columns:
        df[column] = df[column].astype(datatype)
    return df


def encode_binary_columns(df, columns, encoding_method='ordinal'):
    """
    Encodes binary columns using the specified encoding method.
    """
    if encoding_method == 'ordinal':
        for col in columns:
            unique_values = df[col].unique()
            value_map = {value: i for i, value in enumerate(unique_values)}
            df[col] = df[col].replace(value_map).astype(int)
    # Add other encoding methods as needed
    return df

def get_numerical_columns(df):
    """
    Returns a list of column names for numerical columns.
    """
    numerical_columns = list(df.select_dtypes(include=[np.number]).columns)
    return numerical_columns

def get_categorical_columns(df):
    """
    Returns a list of column names containing categorical data in the given DataFrame.
    """
    object_columns = df.select_dtypes(include=['object']).columns.to_list()
    boolean_columns = df.select_dtypes(include=['bool']).columns.to_list()
    category_columns = df.select_dtypes(include=['category']).columns.to_list()
    categorical_columns = object_columns + boolean_columns + category_columns
    return categorical_columns

def get_features(df):
    """
    Returns the first n column names of the DataFrame.
    """
    return df.columns[:n]

# In[ ]:

def drop_missing_data(df, threshold):
    
    #Drops rows from the dataframe where the percentage of missing values is above the thre
    #Args:
    #df (pd.DataFrame): the dataframe to process.
    #threshold (float): the percentage threshold above which rows will be dr
    #Returns:
    #pd.DataFrame: the processed dataframe.
    
    # Calculate the number of non-missing values in each row.
    num_non_missing = df.count(axis=1)
    
    # Calculate the percentage of missing values in each row.
    pct_missing = 1 - num_non_missing / len(df.columns)
 
    # Drop rows where the percentage of missing values is above the threshold.
    df = df.loc[pct_missing <= threshold]
 
    #df = drop_missing_data(df, 0.5)
    return df

def drop_rows_with_high_nans(df, threshold=0.75):
    """
    Drops rows from the dataframe where the percentage of missing values is above the threshold.

    Args:
        df (pd.DataFrame): the dataframe to process.
        threshold (float): the percentage threshold above which rows will be dropped.

    Returns:
        pd.DataFrame: the processed dataframe.
    """
    # Calculate the number of non-missing values in each row.
    num_non_missing = df.count(axis=1)
    
    # Calculate the percentage of missing values in each row.
    pct_missing = 1 - num_non_missing / len(df.columns)
 
    # Drop rows where the percentage of missing values is above the threshold.
    df = df.loc[pct_missing <= threshold]
    
    return df

# Example usage:
#data_clean = drop_rows_with_high_nans(data)


def drop_all_null(df):
    df = df.dropna()
    df = df.dropna(how='all')
    return df

def drop_all_nulls(df, threshold):
    df = df.dropna()
    return df


def remove_outliers_iqr(df, multiplier=1.5):
    # Select only numeric columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Loop through numeric columns
    for col in num_cols:
        # Calculate the IQR of the column
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define the upper and lower bounds for outliers
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        # Filter out the outliers from the DataFrame
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

def remove_outliers_iqr_loop(df, multiplier=1.5):
    # Select only numeric columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Initialize the number of outliers
    num_outliers = 1
    
    while num_outliers > 0:
        # Initialize the number of outliers
        num_outliers = 0
        
        # Loop through numeric columns
        for col in num_cols:
            # Calculate the IQR of the column
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define the upper and lower bounds for outliers
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            # Filter out the outliers from the DataFrame
            num_outliers += df[col][(df[col] < lower_bound) | (df[col] > upper_bound)].count()
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

def standardize_text(df, columns):
    for col in columns:
        df[col] = df[col].str.upper().str.strip()
    return df


def scale_x_data(df):
    # Create an instance of MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler to the 'county' column
    scaler.fit(df[['total_sqft']])

    # Transform the 'county' column using the fitted scaler
    df['total_sqft_scaled'] = scaler.transform(df[['total_sqft']])
  
    return df

def scale_y_data(df):
    # Create an instance of MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler to the 'county' column
    scaler.fit(df[['assessed_property_value']])

    # Transform the 'county' column using the fitted scaler
    df['assessed_property_value_scaled'] = scaler.transform(df[['assessed_property_value']])
    return df

from sklearn.preprocessing import MinMaxScaler

def one_hot_encode_data(df):
    # One-hot encode categorical features
    categorical_features = ['BusinessTravel','Department','EducationField', 'Gender',         'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Reset indexes for both original and encoded DataFrames
    df = df.reset_index(drop=True)
    df_encoded = df_encoded.reset_index(drop=True)

    return df_encoded


def scale_datas(X):
    # Create an instance of MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler to the features
    scaler.fit(X)

    # Transform the features using the fitted scaler
    X_scaled = X.copy()
    X_scaled[X.columns] = scaler.transform(X)

    return X_scaled

def scale_data(df):
    # Create an instance of MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler to all features except the target variable
    scaler.fit(df.drop(['assessed_property_value'], axis=1))

    # Transform all features except the target variable using the fitted scaler
    df[df.drop(['assessed_property_value'], axis=1).columns] = scaler.transform(df.drop(['assessed_property_value'], axis=1))

    return df

def define_X_y(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    return X, y

#X, y = define_X_y(df, 'assessed_property_value')
#X = X[['total_sqft', 'lot_sqft', 'num_rooms', 'year_built']]
def split_datasets(X, y):
    from sklearn.model_selection import train_test_split

    # split the data into train (80%) and a temporary set (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

    # split the temporary set into validate (10% of the original data) and test sets (10% of the original data)
    X_validate, X_test, y_validate, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_validate, X_test, y_train, y_validate, y_test

def split_dataset(df):
    from sklearn.model_selection import train_test_split
    
    # Separate the features (X) and target variable (y)
    X = df.drop(columns=['assessed_property_value_scaled'])
    y = df['assessed_property_value_scaled']

    # split the data into train (80%) and a temporary set (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

    # split the temporary set into validate (10% of the original data) and test sets (10% of the original data)
    X_validate, X_test, y_validate, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_validate, X_test, y_train, y_validate, y_test

# Call the function
#X_train, X_validate, X_test, y_train, y_validate, y_test = wrangle.split_dataset(df)

# Check the shape of X_train
#print(X_train.shape)


from sklearn.model_selection import train_test_split

def split_data(df):
   
    # Separate the features (X) and the target variable (y)
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    # split the data into train (80%) and a temporary set (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

    # split the temporary set into validate (10% of the original data) and test sets (10% of the original data)
    X_validate, X_test, y_validate, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Concatenate the split data back into DataFrames
    train_df = pd.concat([X_train, y_train], axis=1)
    validate_df = pd.concat([X_validate, y_validate], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Combine the train, validate, and test DataFrames into a single DataFrame
    df = pd.concat([train_df, validate_df, test_df], axis=0)

    return df


def drop_column(df, column_name):
    df = df.drop(column_name, axis=1)
    return df

def change_dtype(df, column_name, dtype):
    df[column_name] = df[column_name].astype(dtype)
    return df

def prepare_data(df):
    """
    Prepare the feature matrix X and the target variable y from the given DataFrame.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the property data.
    
    Returns:
    X (pd.DataFrame): The feature matrix containing the selected features.
    y (pd.Series): The target variable, assessed property value.
    """
    X = df[['total_sqft', 'ensuite_bathrooms', 'num_rooms', 'year_built', 'num_full_baths', 'lot_sqft']]
    y = df['assessed_property_value']
    
    # Concatenate X and y into a single DataFrame
    df = pd.concat([X, y], axis=1)
    
    return df

def scaler_data(df):
    # Create an instance of MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler to the features (excluding the target variable)
    scaler.fit(df.drop(['assessed_property_value'], axis=1))

    # Transform all features except the target variable using the fitted scaler
    transformed_data = scaler.transform(df.drop(['assessed_property_value'], axis=1))

    print("Original DataFrame shape:", df.drop(['assessed_property_value'], axis=1).shape)
    print("Transformed DataFrame shape:", transformed_data.shape)

    # Set the transformed data to the original DataFrame
    df[df.drop(['assessed_property_value'], axis=1).columns] = transformed_data

    return df

def clean_up_X_y(df):
    y = df['assessed_property_value']
    df = df.drop('assessed_property_value', axis=1)
    return df

def prepared_data(df):
    """
    Prepare the feature matrix X and the target variable y from the given DataFrame.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the property data.
    
    Returns:
    df (pd.DataFrame): The original DataFrame.
    X (pd.DataFrame): The feature matrix containing the selected features.
    y (pd.Series): The target variable, assessed property value.
    """
    X = df[['total_sqft', 'zip_code', 'county', 'home_type', 'city','num_rooms', 'year_built', 'num_full_baths', 'lot_sqft']]
    y = df['assessed_property_value']
    
    return df

def make_sample(df, frac=0.003):
    """
    This function takes a DataFrame as input and returns a random sample of the data.
    
    Parameters
    ----------
    df : DataFrame
        The input DataFrame to take a random sample from.
    frac : float, optional
        The fraction of the data to take as a sample. Default is 0.001 (0.1%).
        
    Returns
    -------
    DataFrame
        The random sample of the input DataFrame.
    """
    sample_df = df.sample(frac=frac)
    return sample_df

# Example usage
# sample_df = make_sample(df, frac=0.001)


def count_outliers_iqr(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = ((df < lower_bound) | (df > upper_bound)).sum().sum()
    return outliers

def regression_errors(y_true, y_pred):
    # Calculate the mean squared error
    mse = np.mean((y_pred - y_true) ** 2)

    # Calculate the root mean squared error
    rmse = np.sqrt(mse)

    # Calculate the explained variance score
    ssr = np.sum((y_pred - np.mean(y_true)) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    evs = 1 - (ssr / sst)

    # Calculate the R-squared score
    r2 = 1 - (np.sum((y_pred - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

    # Return the metrics as a dictionary
    return {'mse': mse, 'rmse': rmse, 'evs': evs, 'r2': r2}

def select_top_features(df, k=2):
    from sklearn.feature_selection import SelectKBest, f_regression
    
    # Select the top k features for predicting tip amount
    selector = SelectKBest(f_regression, k=k)
    y = df['actual_productivity']
    X_kbest = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]

    # Print the top k features
    print('Top', k, 'Features:', list(selected_features))

    # Return the top k features as a list
    return list(selected_features)

def cv_evaluation(df):
    # Prepare your data
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    # Split your data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale your features, if necessary
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create a list of models to test
    models = [
        ('Linear Regression', LinearRegression()),
        ('Lasso', Lasso()),
        ('Ridge', Ridge()),
        ('ElasticNet', ElasticNet()),
        ('Decision Tree', DecisionTreeRegressor()),
        ('Random Forest', RandomForestRegressor()),
        ('Gradient Boosting', GradientBoostingRegressor()),
        ('XGBoost', XGBRegressor())
    ]

    # Loop through the models and evaluate each one using cross-validation
    for name, model in models:
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5,                                         scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)
        print(f"{name}: Avg. RMSE: {np.mean(rmse_scores):.4f}, Std. Dev: {np.std(rmse_scores):.4f}")

def check_feature_importance(df):
    X = df.drop('actual_productivity', axis=1)
    y = df['actual_productivity']

    model = LinearRegression()
    model.fit(X, y)

    # Pair the column names with their respective coefficients
    feature_importances = list(zip(X.columns, model.coef_))

    # Print the feature importances
    for feature, importance in feature_importances:
        print(f"{feature}: {importance}")

    #return feature_importances

def feature_elimination(df, n_features_to_select=6):
    
    X = df.drop('actual_productivity', axis=1)
    y = df['actual_productivity']
    
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=n_features_to_select)
    rfe.fit(X, y)

    # Get the column names for the features that RFE has selected
    selected_columns = X.columns[rfe.support_]

    # Print the selected columns
    print("Selected columns:", selected_columns)
    
    #return selected_columns
    
    
def hist_frequency_chart(df, continuous_vars=['total_sqft', 'year_built', 'lot_sqft', 'num_rooms']):
    for var in continuous_vars:
        plt.figure()
        plt.hist(df[var], bins=20)
        plt.title(f'Histogram of {var}')
        plt.xlabel(var)
        plt.ylabel('Frequency')
        plt.show()
        
def get_stats(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X = sm.add_constant(X)  # Add a constant term for the intercept
    OLS = sm.OLS(y, X).fit()
    print(OLS.summary())
    return OLS

def model_selection(df, target_col):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    models = [
        ('Linear Regression', LinearRegression()),
        ('Lasso', Lasso()),
        ('Ridge', Ridge()),
        ('ElasticNet', ElasticNet()),
        ('Decision Tree', DecisionTreeRegressor()),
        ('Random Forest', RandomForestRegressor()),
        ('Gradient Boosting', GradientBoostingRegressor()),
        ('XGBoost', XGBRegressor())
    ]

    for name, model in models:
        mse = -np.mean(cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5))
        r2 = np.mean(cross_val_score(model, X, y, scoring='r2', cv=5))
        print(f"{name}: Mean Squared Error = {mse:.4f}, R^2 Score = {r2:.4f}")
        
        
def model_eval(df, target_column):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor

    feature_columns = [col for col in df.columns if col != target_column]

    # Split the data into train (70%), validation (15%), and test (15%) sets
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    validate, test = train_test_split(temp, test_size=0.5, random_state=42)

    # Separate the features and target for each dataset
    X_train, y_train = train[feature_columns], train[target_column]
    X_validate, y_validate = validate[feature_columns], validate[target_column]
    X_test, y_test = test[feature_columns], test[target_column]

    models = [
        ('Linear Regression', LinearRegression()),
        ('Lasso', Lasso()),
        ('Ridge', Ridge()),
        ('ElasticNet', ElasticNet()),
        ('Decision Tree', DecisionTreeRegressor()),
        ('Random Forest', RandomForestRegressor())
    ]

    def get_p_values(X, y, model):
        n = X.shape[0]
        p = X.shape[1]
        y_pred = model.predict(X)
        residuals = y - y_pred
        RSS = np.sum(residuals**2)
        MSE = RSS / (n - p - 1)

        se_beta = np.sqrt(np.diagonal(MSE * np.linalg.inv(np.dot(X.T, X))))
        t_stat = model.coef_ / se_beta
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - p - 1))

        return p_values

    best_model = None
    best_score = -np.inf

    for name, model in models:
        model.fit(X_train, y_train)
        score = model.score(X_validate, y_validate)
        if score > best_score:
            best_score = score
            best_model = (name, model)

    print(f"Best Model: {best_model[0]}, R^2 score: {best_score:.3f}")

   # for feature in feature_columns:
        #sns.regplot(x=feature, y=target_column, data=df)
        #plt.title(f"{feature} vs. {target_column}")
        #plt.show()

    p_values = get_p_values(X_train, y_train, best_model[1])
    for feature, p_value in zip(feature_columns, p_values):
        print(f"P>|t| for {feature}: {p_value:.3f}")
        
def model_evals(df_train, target_column, df_validation):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor

    feature_columns = [col for col in df_validation.columns if col != target_column]

    # Use the validation set
    X_val, y_val = df_validation[feature_columns], df_validation[target_column]

    models = [
        ('Linear Regression', LinearRegression()),
        ('Lasso', Lasso()),
        ('Ridge', Ridge()),
        ('ElasticNet', ElasticNet()),
    ]

    best_model = None
    best_score = -np.inf

    for name, model in models:
        model.fit(X_val, y_val)
        score = model.score(X_val, y_val)  # Evaluate the model on the validation set
        if score > best_score:
            best_score = score
            best_model = (name, model)

    print(f"Best Model: {best_model[0]}, R^2 score: {best_score:.3f}")

    for feature in feature_columns:
        sns.regplot(x=feature, y=target_column, data=df_validation)
        plt.title(f"{feature} vs. {target_column}")
        plt.show()

        
def model_interpretation(df, target_column, best_params):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor

    
    best_params = {
    'n_estimators': 200,
    'max_depth': 30,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'bootstrap': True
    }
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(**best_params)

    rf.fit(X_train, y_train)

    importances = rf.feature_importances_

    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})

    feature_importances = feature_importances.sort_values(by='importance', ascending=False)

    print(feature_importances)


def feature_importance_bar_chart(feature_importances):

    plt.figure(figsize=(10, 5))
    plt.bar(feature_importances['feature'], feature_importances['importance'])
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.xticks(rotation=45)
    plt.show()
    
def model_validation(df, target_column):
    feature_columns = [col for col in df.columns if col != target_column]

    X_train, X_test, y_train, y_test = train_test_split(df[feature_columns], df[target_column], test_size=0.3, random_state=42)

    models = [
        ('Linear Regression', LinearRegression()),
        ('Lasso', Lasso()),
        ('Ridge', Ridge()),
        ('ElasticNet', ElasticNet()),
        ('Decision Tree', DecisionTreeClassifier()),
        ('Random Forest', RandomForestClassifier())
    ]

    def get_p_values(X, y, model):
        n = X.shape[0]
        p = X.shape[1]
        y_pred = model.predict(X)
        residuals = y - y_pred
        RSS = np.sum(residuals**2)
        MSE = RSS / (n - p - 1)

        se_beta = np.sqrt(np.diagonal(MSE * np.linalg.inv(np.dot(X.T, X))))
        t_stat = model.coef_ / se_beta
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - p - 1))

        return p_values

    best_model = None
    best_score = -np.inf

    for name, model in models:
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_model = (name, model)

    print(f"Best Model: {best_model[0]}, R^2 score: {best_score:.3f}")

    p_values = get_p_values(X_train, y_train, best_model[1])
    for feature, p_value in zip(feature_columns, p_values):
        print(f"P>|t| for {feature}: {p_value:.3f}")
        
        
def feature_correlation(df):
    # Remove 'assessed_property_value' column from the list of features
    features = [column for column in df.columns if column != 'actual_productivity']

    # Set the significance level
    alpha = 0.05

    # Loop through all features and perform hypothesis test
    for feature in features:
        X = df[feature]
        y = df['actual_productivity']

        # Calculate Pearson correlation coefficient and p-value
        correlation_coefficient, p_value = stats.pearsonr(X, y)
        correlation_coefficient_s, p_value_s = stats.spearmanr(X, y)

        # Print the results
        print(f"Feature: {feature}")
        print(f"Pearson correlation coefficient: {correlation_coefficient}")
        print(f"P-value: {p_value}")
        
        print("\n")
        print(f"Spearman correlation coefficient: {correlation_coefficient_s}")
        print(f"P-value: {p_value_s}")

        # Test the null hypothesis
        if p_value < alpha:
            print("\033[32mReject the null hypothesis (H0): There is a linear relationship'.\033[0m")
        else:
            print("\033[31mFail to reject the null hypothesis (H0): There is no evidence of a linear relationship'.\033[0m")
        
        print("\n")
        
        
def data_dict():
    data_dict = {
        'id': 'Unique identifier for each property',
        'parcelid': 'Unique identifier for each property, used in conjunction with "assessmentyear" to form a composite primary key',
        'bedroomcnt': 'Number of bedrooms in the property',
        'bathroomcnt': 'Number of bathrooms in the property',
        'fireplacecnt': 'Number of fireplaces in the property',
        'calculatedbathnbr': 'Number of bathrooms in the property (including fractional bathrooms)',
        'calculatedfinishedsquarefeet': 'Total finished living area of the property, in square feet',
        'fullbathcnt': 'Number of full bathrooms in the property (including fractional bathrooms)',
        'garagecarcnt': 'Number of cars that can fit in the garage, if applicable',
        'garagetotalsqft': 'Total square footage of the garage, if applicable',
        'latitude': 'Latitude of the property',
        'longitude': 'Longitude of the property',
        'lotsizesquarefeet': 'Total area of the lot, in square feet',
        'regionidzip': 'Zip code of the property',
        'taxvaluedollarcnt': 'Total tax assessed value of the property, in dollars',
        'roomcnt': 'Total number of rooms in the property (including bedrooms and bathrooms)',
        'yearbuilt': 'Year the property was built',
        'numberofstories': 'Number of stories in the property, if applicable',
        'assessmentyear': 'Year of the property assessment, used in conjunction with "parcelid" to form a composite primary key',
        'landtaxvaluedollarcnt': 'Tax assessed value of the land, in dollars',
        'structuretaxvaluedollarcnt': 'Tax assessed value of the structure, in dollars',
        'taxamount': 'Total property tax for the assessment year, in dollars'
    }

    data_dict_df = pd.DataFrame.from_dict(data_dict, orient='index', columns=['Description'])
    return data_dict_df

def split_and_evaluate_ols(df, target_column):
    from sklearn.model_selection import train_test_split

    # Split the data into train (70%), validation (15%), and test (15%) sets
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    validate, test = train_test_split(temp, test_size=0.5, random_state=42)

    # Evaluate the OLS model
    wrangle.evaluate_ols(train, validate, test, target_column)
    
    
def evaluate_ols_with_splits(df, target_column):
    # Split the data into train (70%), validation (15%), and test (15%) sets
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    validate, test = train_test_split(temp, test_size=0.5, random_state=42)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def evaluate_ols(train, validate, test, target_column):
    feature_columns = [col for col in train.columns if col != target_column]

    X_train, y_train = train[feature_columns], train[target_column]
    X_validate, y_validate = validate[feature_columns], validate[target_column]
    X_test, y_test = test[feature_columns], test[target_column]

    model = LinearRegression()
    model.fit(X_train, y_train)

    def evaluate_set(name, X, y):
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        print(f"{name} R^2 Score: {r2:.3f}")
        print(f"{name} Mean Squared Error: {mse:.3f}")
        return r2

    train_r2 = evaluate_set("Train", X_train, y_train)
    validate_r2 = evaluate_set("Validation", X_validate, y_validate)
    test_r2 = evaluate_set("Test", X_test, y_test)

    print(f"Difference in R^2 Scores:")
    print(f"Train-Validation: {abs(train_r2 - validate_r2):.3f}")
    print(f"Train-Test: {abs(train_r2 - test_r2):.3f}")
    print(f"Validation-Test: {abs(validate_r2 - test_r2):.3f}")
    
def final_split_data(df, target_column):
    # Split the data into train (70%), validation (15%), and test (15%) sets
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    validate, test = train_test_split(temp, test_size=0.5, random_state=42)

    return train, validate, test

def custom_describe_encoded(df):
    desc = df.describe(include='all').T

    # Adding additional columns
    desc['count_nulls'] = df.isnull().sum()
    desc['pct_nulls'] = (desc['count_nulls'] / len(df)) * 100
    desc['num_rows_missing'] = len(df) - desc['count']
    desc['pct_rows_missing'] = (desc['num_rows_missing'] / len(df)) * 100
    desc['dtype'] = df.dtypes

    # Add results from sub-functions to the desc DataFrame
    desc['distribution_type'] = df.apply(distribution_type)
    desc['skewness'] = df.apply(column_skewness)
    desc['skew_type'] = df.apply(skew_type)
    desc['data_type'] = df.apply(data_type)
    desc['num_outliers'] = df.apply(iqr_outliers)
    desc['variable_type'] = df.apply(variable_type)

    # Calculate correlations between numeric columns
    correlations = df.corr().abs().unstack().sort_values(ascending=False).drop_duplicates()
    correlations = correlations[(correlations != 1) &                                                     correlations.index.get_level_values(0).equals(correlations.index.get_level_values(1))]

    # Reorder columns
    columns_order = [
        'num_rows_missing', 'pct_rows_missing', 'count', 'count_nulls', 'pct_nulls', 'mean', 'std',           'min', '25%', '50%', '75%', 'max', 'unique', 'top', 'freq', 'dtype', 'distribution_type',             'skewness', 'skew_type', 'data_type', 'num_outliers', 'variable_type'
    ]
    desc = desc[columns_order]

    # Display the custom describe DataFrame with left-aligned column names
    display(desc.style.set_properties(**{'text-align': 'left'}))
    
    return desc


def custom_describe(df):
    desc = df.describe(include='all').T

    # Adding additional columns
    desc['count_nulls'] = df.isnull().sum()
    desc['pct_nulls'] = (desc['count_nulls'] / len(df)) * 100
    desc['num_rows_missing'] = len(df) - desc['count']
    desc['pct_rows_missing'] = (desc['num_rows_missing'] / len(df)) * 100
    desc['dtype'] = df.dtypes

    # Add results from sub-functions to the desc DataFrame
    desc['distribution_type'] = df.apply(distribution_type)
    desc['skewness'] = df.apply(column_skewness)
    desc['skew_type'] = df.apply(skew_type)
    desc['data_type'] = df.apply(data_type)
    desc['num_outliers'] = df.apply(iqr_outliers)
    desc['variable_type'] = df.apply(variable_type)

    # Calculate correlations between numeric columns
    correlations = df.corr().abs().unstack().sort_values(ascending=False).drop_duplicates()
    correlations = correlations[(correlations != 1) &                                                     correlations.index.get_level_values(0).equals(correlations.index.get_level_values(1))]

    
    for col in df.columns:
        print("\nColumn:", col)
        
        if df[col].dtype != 'O':
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.show()

    # Reorder columns
    columns_order = [
    'num_rows_missing', 'pct_rows_missing', 'count', 'count_nulls', 'pct_nulls', 'mean', 'std',
    'min', '25%', '50%', '75%', 'max', 'unique', 'top', 'freq', 'dtype', 'distribution_type',
    'skewness', 'skew_type', 'data_type', 'num_outliers', 'variable_type'
]

    # Filter the columns_order list to only include columns present in the desc DataFrame
    columns_order = [col for col in columns_order if col in desc.columns]

    # Reorder the columns in the desc DataFrame
    desc = desc[columns_order]

    return desc

def custom_describex(df):
    desc = df.describe(include='all').T

    # Adding additional columns
    desc['count_nulls'] = df.isnull().sum()
    desc['pct_nulls'] = (desc['count_nulls'] / len(df)) * 100
    desc['num_rows_missing'] = len(df) - desc['count']
    desc['pct_rows_missing'] = (desc['num_rows_missing'] / len(df)) * 100
    desc['dtype'] = df.dtypes

    # Add results from sub-functions to the desc DataFrame
    desc['distribution_type'] = df.apply(distribution_type)
    desc['skewness'] = df.apply(column_skewness)
    desc['skew_type'] = df.apply(skew_type)
    desc['data_type'] = df.apply(data_type)
    desc['num_outliers'] = df.apply(iqr_outliers)
    desc['variable_type'] = df.apply(variable_type)

    # Replace "N/A" values with None
    desc.replace("N/A", np.nan, inplace=True)

    # Calculate correlations between numeric columns
    correlations = df.corr().abs().unstack().sort_values(ascending=False).drop_duplicates()
    correlations = correlations[(correlations != 1) &                     correlations.index.get_level_values(0).equals(correlations.index.get_level_values(1))]

    
    for col in df.columns:
        print("\nColumn:", col)
        
        if df[col].dtype != 'O':
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.show()

    # Reorder columns
    columns_order = [
        'num_rows_missing', 'pct_rows_missing',
        'count', 'count_nulls', 'pct_nulls', 'mean', 'std', 'min', '25%', '50%', '75%', 'max',                 'unique', 'top', 'freq', 'dtype',
        'distribution_type', 'skewness', 'skew_type', 'data_type', 'num_outliers', 'variable_type'
    ]
    desc = desc[columns_order]
    
    return desc

def distribution_type(column):
    if not np.issubdtype(column.dtype, np.number):
        return None
    
    _, p_value = stats.normaltest(column.dropna())
    return 'Normal' if p_value > 0.05 else 'Non-normal'


def column_skewness(column):
    if not np.issubdtype(column.dtype, np.number):
        return None
    return stats.skew(column.dropna())

def skew_type(column):
    skew = column_skewness(column)
    if skew is None:
        return None
    if skew > 0:
        return 'Right'
    elif skew < 0:
        return 'Left'
    else:
        return 'Symmetric'

def data_type(column):
    if not np.issubdtype(column.dtype, np.number):
        return None
    unique_count = column.nunique()
    if unique_count / len(column) < 0.05:
        return 'Discrete'
    else:
        return 'Continuous'
    
def iqr_outliers(column):
    if not np.issubdtype(column.dtype, np.number):
        return None
    
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((column < lower_bound) | (column > upper_bound)).sum() 
    
def variable_type(column):
    num_unique_values = len(column.unique())
    
    if column.dtype == np.object:
        if num_unique_values <= 10:
            return 'nominal'
        else:
            return 'categorical'
    elif column.dtype == np.int64 or column.dtype == np.float64:
        if num_unique_values <= 10:
            return 'ordinal'
        else:
            return 'numerical'
    else:
        return 'unknown'
    
def handle_missing_values(df, prop_required_column, prop_required_row):
    # Drop columns based on the proportion of missing values
    column_threshold = int(round(prop_required_column * len(df)))
    df = df.dropna(axis=1, thresh=column_threshold)

    # Drop rows based on the proportion of missing values
    row_threshold = int(round(prop_required_row * len(df.columns)))
    df = df.dropna(axis=0, thresh=row_threshold)

    return df
    
def recommend_visualizations(correlations):
    recommendations = []

    for corr in correlations:
        if abs(corr) >= 0.7:
            recommendations.append("scatter plot")
        elif abs(corr) >= 0.3:
            recommendations.append("line chart")
        else:
            recommendations.append("bar chart")

    return ", ".join(set(recommendations))