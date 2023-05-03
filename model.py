#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def plot_and_evaluate(actual, predicted, model):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.legend()
    plt.title(f"{model} - Actual vs Predicted")
    plt.show()

    return mse, rmse


# In[ ]:


# Last Observed Value
def last_observed_value(train, test):
    last_value = train.iloc[-1]
    predictions = np.full(test.shape, last_value)

    mse, rmse = plot_and_evaluate(test, predictions, 'Last Observed Value')
    return mse, rmse

mse, rmse = last_observed_value(train, test)
eval_df = eval_df.append({'model': 'Last Observed Value', 'metric': 'MSE', 'value': mse}, ignore_index=True)
eval_df = eval_df.append({'model': 'Last Observed Value', 'metric': 'RMSE', 'value': rmse}, ignore_index=True)


# In[ ]:


# Simple Average
def simple_average(train, test):
    avg = train.mean()
    predictions = np.full(test.shape, avg)

    mse, rmse = plot_and_evaluate(test, predictions, 'Simple Average')
    return mse, rmse

mse, rmse = simple_average(train, test)
eval_df = eval_df.append({'model': 'Simple Average', 'metric': 'MSE', 'value': mse}, ignore_index=True)
eval_df = eval_df.append({'model': 'Simple Average', 'metric': 'RMSE', 'value': rmse}, ignore_index=True)


# In[ ]:


# Moving Average
def moving_average(train, test, window):
    predictions = train.rolling(window=window).mean().iloc[-1]
    mse, rmse = plot_and_evaluate(test, predictions, 'Moving Average')
    return mse, rmse

mse, rmse = moving_average(train, test, 4)  # Using a 4-period moving average
eval_df = eval_df.append({'model': 'Moving Average', 'metric': 'MSE', 'value': mse}, ignore_index=True)
eval_df = eval_df.append({'model': 'Moving Average', 'metric': 'RMSE', 'value': rmse}, ignore_index=True)


# In[ ]:


# Holt's Linear Trend Model
def holts_linear_trend(train, test):
    model = Holt(train).fit(smoothing_level=0.8, smoothing_trend=0.2)
    predictions = model.forecast(len(test))

    mse, rmse = plot_and_evaluate(test, predictions, "Holt's Linear Trend Model")
    return mse, rmse

mse, rmse = holts_linear_trend(train, test)
eval_df = eval_df.append({'model': "Holt's Linear Trend Model", 'metric': 'MSE', 'value': mse}, ignore_index=True)
eval_df = eval_df.append({'model': "Holt's Linear Trend Model", 'metric': 'RMSE', 'value': rmse}, ignore_index=True)

