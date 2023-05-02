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
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
# In[ ]:


def avg_team_ot(df):
    # Calculate the average overtime across all teams
    avg_overtime = df['over_time'].mean()

    # Find the three teams with the most overtime
    top_3_overtime_teams = df.groupby('team')['over_time'].mean().sort_values(ascending=False).head(3).index

    # Create a new column to indicate if the team is in the top 3
    df['highlight'] = df['team'].apply(lambda x: 'Top 3' if x in top_3_overtime_teams else 'Others')

    plt.figure(figsize=(10, 6))

    # Plot the bar chart, highlighting the top 3 teams
    sns.barplot(x='team', y='over_time', data=df, hue='highlight', palette={'Top 3': 'gray', 'Others': 'blue'}, dodge=False)

    # Add the average overtime h-line
    plt.axhline(avg_overtime, color='green', linestyle='--', label='Average Overtime')

    plt.title('Overtime by Team')
    plt.xlabel('Team')
    plt.ylabel('Overtime (minutes)')
    plt.legend(title='Team Category', loc='upper right')

    plt.show()

def incentive_over_time(df):
    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Calculate the interquartile range (IQR) to detect and remove outliers
    Q1 = df['incentive'].quantile(0.25)
    Q3 = df['incentive'].quantile(0.75)
    IQR = Q3 - Q1

    # Filter out the outliers
    filtered_df = df[(df['incentive'] >= (Q1 - 1.5 * IQR)) & (df['incentive'] <= (Q3 + 1.5 * IQR))]

    # Group the data by week and calculate the average incentive per week
    weekly_incentive = filtered_df.groupby(pd.Grouper(key='date', freq='W'))['incentive'].mean().reset_index()

    # Plot the line chart
    plt.plot(weekly_incentive['date'], weekly_incentive['incentive'])
    plt.xlabel('Date')
    plt.ylabel('Average Incentive')
    plt.title('Incentive Over Time (Per Week)')
    plt.xticks(rotation=45)
    plt.show()
    
    
def plot_production_data(df):
    """
    Plot a bar chart of prod_capacity and actual_production by team, and add a horizontal line
    for the average actual_production.
    Args:
        df: a pandas DataFrame containing production data with columns 'team', 'prod_capacity', and 'actual_production'
    Returns:
        None
    """
    # Calculate the max actual_production
    max_actual_production = df['actual_production'].mean()

    # Group the data by team and calculate the mean of actual_productivity, prod_capacity, and actual_production
    grouped_data = df.groupby('team')[['prod_capacity', 'actual_production']].mean()

    # Plot the bar chart
    grouped_data.plot(kind='bar', figsize=(10, 6))

    # Add a horizontal line for the max actual_production
    plt.axhline(max_actual_production, color='red', linestyle='--', label=f'Average Actual Production: {max_actual_production:.2f}')

    # Set the title and axis labels
    plt.title('Prod Capacity, and Actual Production by Team')
    plt.xlabel('Team')
    plt.ylabel('Value')
    plt.legend()

    # Show the plot
    plt.show()
    
    
def plot_monthly_productivity_by_team(df):
    # Convert the 'date' column to datetime type if not already
    df['date'] = pd.to_datetime(df['date'])

    # Extract the month from the 'date' column
    df['month'] = df['date'].dt.to_period('M')

    # Calculate the average actual_productivity across all teams
    avg_productivity = df['actual_productivity'].mean()

    # Factory productivity standard
    factory_standard = 0.75

    # Get the list of unique months and teams
    unique_months = df['month'].unique()
    unique_teams = df['team'].unique()

    # Create subplots for each month
    n_months = len(unique_months)
    fig, axes = plt.subplots(n_months, 1, figsize=(10, 5 * n_months), sharex=True)

    for i, month in enumerate(unique_months):
        # Filter data for the current month
        month_data = df[df['month'] == month]

        # Calculate the mean actual_productivity for each team in the current month
        team_productivity = month_data.groupby('team')['actual_productivity'].mean()

        # Find the team with the highest actual_productivity
        max_team = team_productivity.idxmax()

        # Set bar colors
        colors = ['tab:gray' if team != max_team else 'tab:orange' for team in unique_teams]

        # Create a bar chart
        axes[i].bar(unique_teams, team_productivity, color=colors)

        # Add horizontal lines
        axes[i].axhline(avg_productivity, color='green', linestyle='--', label='Average Productivity')
        axes[i].axhline(factory_standard, color='red', linestyle='--', label='Factory Standard')

        # Set labels and title
        axes[i].set_xlabel('Team')
        axes[i].set_ylabel('Actual Productivity')
        axes[i].set_title(f'Actual Productivity by Team for {month}')
        axes[i].legend()

    plt.tight_layout()
    plt.show()
    
    
def plot_incentive_per_team(df):
    # Calculate the average incentive per team
    grouped_df = df.groupby('team')['incentive'].mean().reset_index()

    # Calculate the overall average incentive across all teams
    average_incentive = df['incentive'].mean()

    # Find the top 3 teams by their average incentives
    top_3_teams = grouped_df.nlargest(3, 'incentive')['team']

    # Create a custom color list, setting 'red' for top 3 teams and 'blue' for others
    colors = ['grey' if team in top_3_teams.values else 'blue' for team in grouped_df['team']]

    # Plot the bar chart
    plt.bar(grouped_df['team'].astype(str), grouped_df['incentive'], color=colors)
    plt.axhline(average_incentive, color='g', linestyle='--', label='Average Incentive')

    plt.xlabel('Team')
    plt.ylabel('Incentive')
    plt.title('Average Incentive per Team')
    plt.legend()
    plt.show()
    
    
def plot_actual_productivity(df):
    """
    Plots a bar chart of the actual productivity by team, highlighting the team with the highest productivity in blue.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the actual productivity data by team.

    Returns:
    None.
    """

    # Calculate the average actual_productivity across all teams
    average_productivity = df['actual_productivity'].mean()

    # Group the data by team and calculate the average actual_productivity for each team
    grouped_df = df.groupby('team')['actual_productivity'].mean().reset_index()

    # Find the team with the highest actual_productivity
    highest_team = grouped_df.loc[grouped_df['actual_productivity'].idxmax()]['team']

    # Plot the bar chart with team numbers on x-axis and actual_productivity on y-axis
    # Highlight the highest team in blue and others in light blue
    bar_colors = ['blue' if team == highest_team else 'lightblue' for team in grouped_df['team']]
    plt.bar(grouped_df['team'], grouped_df['actual_productivity'], color=bar_colors)

    # Add a horizontal red dashed line representing the average actual_productivity
    plt.axhline(average_productivity, color='red', linestyle='--', label=f'Average: {average_productivity:.2f}')

    # Label the axes and add a title
    plt.xlabel('Team')
    plt.ylabel('Actual Productivity')
    plt.title('Actual Productivity by Team')
    plt.legend()
    plt.show()
    
def plot_average_incentive(df):
    """
    Plots a bar chart of the average incentive per team,
    highlighting the top 3 teams in red and others in blue.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the incentive and team columns.
    """
    # Calculate the average incentive per team
    grouped_df = df.groupby('team')['incentive'].mean().reset_index()

    # Calculate the overall average incentive across all teams
    average_incentive = df['incentive'].mean()

    # Find the top 3 teams by their average incentives
    top_3_teams = grouped_df.nlargest(3, 'incentive')['team']

    # Create a custom color list, setting 'red' for top 3 teams and 'blue' for others
    colors = ['grey' if team in top_3_teams.values else 'blue' for team in grouped_df['team']]

    # Plot the bar chart
    plt.bar(grouped_df['team'].astype(str), grouped_df['incentive'], color=colors)
    plt.axhline(average_incentive, color='g', linestyle='--', label='Average Incentive')

    plt.xlabel('Team')
    plt.ylabel('Incentive')
    plt.title('Average Incentive per Team')
    plt.legend()
    plt.show()
    
    
def plot_actual_productivity_by_team(df):
    # Calculate the average actual productivity across all teams
    avg_actual_productivity = df['actual_productivity'].mean()

    # Find the team with the highest actual productivity
    highest_team = df.loc[df['actual_productivity'].idxmax()]['team']

    # Prepare the data for the bar chart
    team_data = df.groupby('team')['actual_productivity'].mean().reset_index()

    # Create the colormap for the bars
    norm = mcolors.Normalize(vmin=team_data['actual_productivity'].min(), vmax=team_data['actual_productivity'].max())
    cmap = plt.get_cmap('Blues')

    # Create the bar chart
    plt.bar(team_data['team'].astype(str), team_data['actual_productivity'],
            color=cmap(norm(team_data['actual_productivity'].values)))

    # Add horizontal lines for the average actual productivity and the factory productivity standard
    plt.axhline(avg_actual_productivity, color='green', linestyle='--', label='Average Productivity')
    plt.axhline(0.75, color='red', linestyle='--', label='Factory Productivity Standard')

    # Set labels and title
    plt.xlabel('Team')
    plt.ylabel('Actual Productivity')
    plt.title('Actual Productivity by Team')
    plt.legend()

    plt.show()
    
    
def plot_actual_productivity_by_team(df):
    # Calculate the average actual productivity across all teams
    avg_actual_productivity = df['actual_productivity'].mean()

    # Find the team with the highest actual productivity
    highest_team = df.loc[df['actual_productivity'].idxmax()]['team']

    # Prepare the data for the bar chart
    team_data = df.groupby('team')['actual_productivity'].mean().reset_index()

    # Create the colormap for the bars
    norm = mcolors.Normalize(vmin=team_data['actual_productivity'].min(), vmax=team_data['actual_productivity'].max())
    cmap = plt.get_cmap('Blues')

    # Create the bar chart
    plt.bar(team_data['team'].astype(str), team_data['actual_productivity'],
            color=cmap(norm(team_data['actual_productivity'].values)))

    # Add horizontal lines for the average actual productivity and the factory productivity standard
    plt.axhline(avg_actual_productivity, color='green', linestyle='--', label='Average Productivity')
    plt.axhline(0.75, color='red', linestyle='--', label='Factory Productivity Standard')

    # Set labels and title
    plt.xlabel('Team')
    plt.ylabel('Actual Productivity')
    plt.title('Actual Productivity by Team')
    plt.legend()

    plt.show()
    
    
import matplotlib.colors as mcolors

def team_productivity_chart(df):
    # Calculate the average actual productivity across all teams
    avg_actual_productivity = df['actual_productivity'].mean()

    # Find the team with the highest actual productivity
    highest_team = df.loc[df['actual_productivity'].idxmax()]['team']

    # Prepare the data for the bar chart
    team_data = df.groupby('team')['actual_productivity'].mean().reset_index()

    # Create the colormap for the bars
    norm = mcolors.Normalize(vmin=team_data['actual_productivity'].min(), vmax=team_data['actual_productivity'].max())
    cmap = plt.get_cmap('Blues')

    # Create the bar chart
    plt.bar(team_data['team'].astype(str), team_data['actual_productivity'],
            color=cmap(norm(team_data['actual_productivity'].values)))

    # Add horizontal lines for the average actual productivity and the factory productivity standard
    plt.axhline(avg_actual_productivity, color='green', linestyle='--', label='Average Productivity')
    plt.axhline(0.75, color='red', linestyle='--', label='Factory Productivity Standard')

    # Set labels and title
    plt.xlabel('Team')
    plt.ylabel('Actual Productivity')
    plt.title('Actual Productivity by Team')
    plt.legend()

    plt.show()
    
def plot_actual_productivity(df):
    # Calculate the average actual productivity across all teams
    avg_actual_productivity = df['actual_productivity'].mean()

    # Find the team with the highest actual productivity
    highest_team = df.loc[df['actual_productivity'].idxmax()]['team']

    # Prepare the data for the bar chart
    team_data = df.groupby('team')['actual_productivity'].mean().reset_index()

    # Create the colormap for the bars
    norm = mcolors.Normalize(vmin=team_data['actual_productivity'].min(), vmax=team_data['actual_productivity'].max())
    cmap = plt.get_cmap('Blues')

    # Create the bar chart
    plt.bar(team_data['team'].astype(str), team_data['actual_productivity'],
            color=cmap(norm(team_data['actual_productivity'].values)))

    # Add horizontal lines for the average actual productivity and the factory productivity standard
    plt.axhline(avg_actual_productivity, color='green', linestyle='--', label='Average Productivity')
    plt.axhline(0.75, color='red', linestyle='--', label='Factory Productivity Standard')

    # Set labels and title
    plt.xlabel('Team')
    plt.ylabel('Actual Productivity')
    plt.title('Actual Productivity by Team')
    plt.legend()

    plt.show()
    
    
def plot_avg_incentive_per_team(df):
    # Calculate the average incentive per team
    grouped_df = df.groupby('team')['incentive'].mean().reset_index()

    # Calculate the overall average incentive across all teams
    average_incentive = df['incentive'].mean()

    # Plot the bar chart
    plt.bar(grouped_df['team'].astype(str), grouped_df['incentive'])
    plt.axhline(average_incentive, color='r', linestyle='--', label='Average Incentive')

    plt.xlabel('Team')
    plt.ylabel('Incentive')
    plt.title('Average Incentive per Team')
    plt.legend()
    plt.show()
    
def productivity_by_team(df):
    # Calculate the average actual productivity
    avg_actual_productivity = df['actual_productivity'].mean()

    # Find the top 3 teams with highest actual productivity
    top_3_actual_productivity_teams = df.groupby('team')['actual_productivity'].mean().sort_values(ascending=False).head(3).index

    # Create a new column to indicate if the team is in the top 3
    df['highlight'] = df['team'].apply(lambda x: 'Top 3' if x in top_3_actual_productivity_teams else 'Others')

    # Set the color palette for the chart
    palette = {'Top 3': 'darkgrey', 'Others': 'lightgrey'}

    # Create the bar plot, highlighting the top 3 teams
    sns.barplot(x='team', y='actual_productivity', data=df, hue='highlight', palette=palette, dodge=False)

    # Add the average actual productivity h-line
    plt.axhline(avg_actual_productivity, color='green', linestyle='--', label=f'Average Actual Productivity: {avg_actual_productivity:.2f}')

    # Set the title and axis labels
    plt.title('Actual Productivity by Team')
    plt.xlabel('Team')
    plt.ylabel('Actual Productivity')

    # Rotate the x-axis labels 90 degrees
    plt.xticks(rotation=90)

    # Show the legend
    plt.legend(title='Team Category', loc='upper right')

    # Show the plot
    plt.show()
    
  
    
'''def plot_resampled_data(data, title, chart_type='line'):
    weekly_filtered = weekly[['over_time', 'wip']]
    plt.figure(figsize=(12, 6))
    
    if chart_type == 'line':
        sns.lineplot(data=data)
    elif chart_type == 'bar':
        sns.barplot(data=data)
        
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.show()

# Filter the columns to include only 'over_time', 'wip', 'idle_time', and 'no_of_workers'
weekly_filtered = weekly[['over_time', 'wip']]

# Plot weekly resampled data with line chart
plot_resampled_data(weekly_filtered, 'Weekly Data', chart_type='line')'''

def plot_overtime(df):
    # Calculate the average overtime across all teams
    avg_overtime = df['over_time'].mean()

    # Find the three teams with the most overtime
    top_3_overtime_teams = df.groupby('team')['over_time'].mean().sort_values(ascending=False).head(3).index

    # Create a new column to indicate if the team is in the top 3
    df['highlight'] = df['team'].apply(lambda x: 'Top 3' if x in top_3_overtime_teams else 'Others')

    plt.figure(figsize=(10, 6))

    # Plot the bar chart, highlighting the top 3 teams
    sns.barplot(x='team', y='over_time', data=df, hue='highlight', palette={'Top 3': 'gray', 'Others': 'blue'}, dodge=False)

    # Add the average overtime h-line
    plt.axhline(avg_overtime, color='green', linestyle='--', label='Average Overtime')

    plt.title('Overtime by Team')
    plt.xlabel('Team')
    plt.ylabel('Overtime (minutes)')
    plt.legend(title='Team Category', loc='upper right')

    plt.show()
    
    
def actual_vs_predicted(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    plt.show()