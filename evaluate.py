#     8888888888 888     888     d8888 888     888     888       d8888 88888888888 8888888888     # 
#     888        888     888    d88888 888     888     888      d88888     888     888            # 
#     888        888     888   d88P888 888     888     888     d88P888     888     888            # 
#     8888888    Y88b   d88P  d88P 888 888     888     888    d88P 888     888     8888888        # 
#     888         Y88b d88P  d88P  888 888     888     888   d88P  888     888     888            # 
#     888          Y88o88P  d88P   888 888     888     888  d88P   888     888     888            # 
#     888           Y888P  d8888888888 888     Y88b. .d88P d8888888888     888     888            # 
#     8888888888     Y8P  d88P     888 88888888 "Y88888P" d88P     888     888     8888888888     # 
#|-----------------------------------------------------------------------------------------------|#
#|-----------------------------------------------------------------------------------------------|#
# IMPORTS
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import math
import sklearn.metrics
#|-----------------------------------------------------------------------------------------------|#
#|-----------------------------------------------------------------------------------------------|#
def plot_residuals(df, yhat, y):
    '''
    This function takes in a dataframe, the actual target variable 
    and model predictions then creates columns for residuals
    and baseline residuals. It returns a graph of both residual columns.
    '''
    # create a residual column
    df['residual'] = (yhat - y)

    # create a residual baseline column
    df['residual_baseline'] = (y.mean() - y)
    
    # set parameters for the plot
    fig, ax = plt.subplots(figsize=(13,7))
    
    ax.hist(df.residual_baseline, label='baseline residuals', alpha=.6)
    ax.hist(df.residual, label='model residuals', alpha=.6)
    ax.legend()
    
    residuals = yhat - y
    plt.scatter(y, residuals)
    plt.axhline(y=0, color='black')
    plt.show()

    plt.show()
#|-----------------------------------------------------------------------------------------------|#    
def plot_residuals2(x, y, yhat):
    """ Plots residuals vs y. The second approach to solving this """
    residuals = yhat - y
    
    baseline = np.full(len(y),y.mean())
    
    baseline_residuals = baseline - y
    
    plt.subplots(2,2, figsize=(16,5))
    plt.subplot(221)
    plt.scatter(x=x, y = residuals)
    plt.axhline(0)
    plt.xlabel('x')
    plt.ylabel('residual (yhat - y)')
    plt.title("OLS Residuals")
    plt.subplot(222)
    plt.scatter(x=x, y = baseline_residuals)
    plt.axhline(0)
    plt.xlabel('x')
    plt.ylabel('residual (yhat - y)')
    plt.title("Baseline residuals")

    plt.subplot(223)
    plt.scatter(x=y, y = residuals)
    plt.axhline(0)
    plt.xlabel('y')
    plt.ylabel('residual (yhat - y)')
    plt.title("OLS Residuals")
    plt.subplot(224)
    plt.scatter(x=y, y = baseline_residuals)
    plt.axhline(0)
    plt.xlabel('y')
    plt.ylabel('residual (yhat - y)')
    plt.title("Baseline residuals")

    plt.tight_layout()
#|-----------------------------------------------------------------------------------------------|#
def regression_errors(df, yhat, y):
    '''
    Returns error metrics for given y and yhat.
    '''
    
    SSE = mean_squared_error(yhat, y)*len(df)
    MSE = mean_squared_error(yhat, y)
    RMSE = sqrt(mean_squared_error(yhat, y))
    ESS = sum((yhat - y.mean())**2)
    TSS = sum((y - y.mean())**2)

    # compute explained variance
    R2 = ESS / TSS
    
    print('SSE is:', SSE)
    print('ESS is:', ESS)
    print('TSS is:', TSS)
    print('R2 is:', R2)
    print('MSE is:', MSE)
    print('RMSE is:', RMSE)
#|-----------------------------------------------------------------------------------------------|#    
def baseline_mean_errors(y):
    """ Compute the SSE, MSE, and RMSE for the baseline model (mean) """
    import sklearn.metrics
    import math
    baseline = y.mean()
    residuals = baseline - y
    residuals_squared = sum(residuals**2)
    SSE = residuals_squared
    print(f'SSE baseline is {SSE}')
    
    MSE = SSE/len(y)
    print(f'MSE baseline is {MSE}')
    
    RMSE = sqrt(MSE)
    print(f'RMSE baseline {RMSE}')
#|-----------------------------------------------------------------------------------------------|#
def better_than_baseline(yhat, y):
     """ Returns True if the model performs better than baseline based on RMSE """
    baseline = y.mean()
    residuals_baseline = baseline - y
    residuals_squared_baseline = sum(residuals_baseline**2)
    SSE_baseline = residuals_squared_baseline
    
    MSE_baseline = SSE_baseline/len(y)
    
    RMSE_baseline = sqrt(MSE_baseline)
    
    residuals = yhat - y
    residuals_squared = sum(residuals**2)
    SSE = residuals_squared
    
    MSE = sklearn.metrics.mean_squared_error(y,yhat)
    
    RMSE = sqrt(sklearn.metrics.mean_squared_error(y,yhat))
    
    if RMSE < RMSE_baseline:
        return True
    else: 
        return False
#|-----------------------------------------------------------------------------------------------|#
#|-----------------------------------------------------------------------------------------------|#