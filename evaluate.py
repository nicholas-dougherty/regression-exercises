import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import math
import sklearn.metrics

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
    
    fig, ax = plt.subplots(figsize=(13,7))

    ax.hist(df.residual_baseline, label='baseline residuals', alpha=.6)
    ax.hist(df.residual, label='model residuals', alpha=.6)
    ax.legend()
    
    residuals = yhat - y
    plt.scatter(y, residuals)
    plt.axhline(y=0, color='black')
    plt.show()

    plt.show()

def regression_errors(df, yhat, y):
    '''
    
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
    
def baseline_mean_errors(y):
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

import sklearn.metrics
import math
def better_than_baseline(yhat, y):
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