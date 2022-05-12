***
```
        ___     ___     ___     ___     ___     ___     ___     ___     ___    _  _   
       | _ \   | __|   / __|   | _ \   | __|   / __|   / __|   |_ _|   / _ \  | \| |  
       |   /   | _|   | (_ |   |   /   | _|    \__ \   \__ \    | |   | (_) | | .` |  
       |_|_\   |___|   \___|   |_|_\   |___|   |___/   |___/   |___|   \___/  |_|\_|  
     _|"""""|_|"""""|_|"""""|_|"""""|_|"""""|_|"""""|_|"""""|_|"""""|_|"""""|_|"""""| 
     "`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-' 
```
***

## Predicting Continuous Outcomes Using Regression

In this repository, we will analyze, visualize and model various labeled datasets with continuous target variables that are being stored in MySQL. This means we will do supervised machine learning (because the data is labeled) using regression (because the target variable we are analyzing is continuous) on structured data (because the data can be naturally stored in rows and columns).
***
### Goals

1. Acquisition-Gather: Gather structured data from SQL to Pandas 
1. Acquisition-Summarize: Summarize the data through aggregates, descriptive stats and distribution plots (histograms, density plots, boxplots, e.g.). (pandas: .value_counts, .head, .shape, .describe, .info, matplotlib.pyplot.hist, seaborn.boxplot) 
1. Preparation-Clean: We will convert datatypes and handle missing values. In this module we will keep it simple in how we handle missing values. We will introduce other ways to handle missing values as we progress through the course. (pandas: .isnull, .value_counts, .dropna, .replace)
1. Preparation-Split: We will sample the data so that we are only using part of our available data to analyze and model. We will discuss the reasons for doing this. This is known as "Train, Validate, Test Splitting". (sklearn.model_selection.train_test_split). 
1. Preparation-Scale: We will discuss the importance of "scaling" data, i.e. putting variables of different units onto the same scale. We will scale data of different units to be on the same scale so they can be compared and modeled. We will discuss different methods for scaling data and why to use one type over another. (sklearn.preprocessing: StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler)
1. Exploration-Hypothesize: We will discuss the meaning of "drivers", variables vs. features, and the target variable. We will disucss the importance of documenting questions and hypotheses, obtaining answers for those questions, and documenting takeaways and findings at each step of exploration.
1. Exploration-Visualize: We will use visualization techniques (scatterplot, jointplot, pairgrid, heatmap) to identify drivers. When a visualization needs to be followed up with a test, we will do so.
1. Exploration-Test: We will analyze the drivers of a continuous variable using appropriate statistical tests (t-tests and correlation tests).
1. Modeling-Feature Engineering: We will learn ways to identify, select, and create features through feature engineering methods, specifically feature importance. We will discuss the "Curse of Dimensionality." (sklearn.feature_selection.f_regression).
1. Modeling-Establish Baseline: We will learn about the importance of establishing a "baseline model" or baseline score and ways to complete this task.
1. Modeling-Build Models: We will build linear regression models, i.e. we will use well established algorithms, such as glm (generalized linear model) or a basic linear regression algorithm (e.g. y = mx + b), to extract the patterns the data is demonstrating and return to us a mathematical model or function (e.g. y = 3x + 2) that will predict the target variable or outcome we want to predict. We will learn about the differences in the most common regression algorithms. (sklearn.linear_model)
1. Modeling-Model Evaluation: We will compare regression models by computing "evaluation metrics", i.e. metrics that measure how well a model did at predicting the target variable. (statsmodels.formula.api.ols, sklearn.metrics, math.sqrt)
1. Modeling-Model Selection and Testing: We will learn how to select a model, and we will test the model on the unseen data sample (the out-of-sample data in the validate and then test datasets).
1. Data Science Pipeline and Product Delivery: We will end with an end-to-end project practicing steps of the data science pipeline from planning through model selection and delivery.
***
### Glossing Over Regression

Regression is a supervised machine learning technique used to model the relationship of one or more features or independent variables, (one = simple regression, more = multiple regression) to one or more target or dependent variables, (one = univariate regression, more = multivariate regression). The variables are represented by continuous data.

A regression algorithm attempts to find the function that best 'mimics' or 'models' the relationship between independent feature(s) and dependent target variable(s). The algorithm does this by finding the line (for simple regression) or plane (for multiple regression) that minimizes the errors in our predictions when compared to the labeled data. Once we acquire that function, it can be used to make predictions on new observations when they become available; we can simply run these new values of the independent variable(s) through the function for each observation to predict the dependent target variable(s).

The algorithm attempts to find the “best” choices of values for the parameters, which in a linear regression model are the coefficients, $bi$, in order to make the formula as “accurate” as possible, i.e. minimize the error. There are different ways to define the error, but whichever evaluation metric we select, the algorithm finds the line of best fit by identifying the parameters that minimize that error.

Once estimated, the parameters (intercept and coefficients) allow the value of the target variable to be obtained from the values of the feature variables.

---
##### Simple Linear Regression

In the simple linear case, our feature is x and our target is y. The algorithm finds the parameters that minimize the error between the actual values and the estimated values. The parameters the algorithm estimates are the slope, β, and the y-intercept, α. ϵ is the error term or the residual value. The residual is the difference of the actual value from the predicted value.

##### Multiple Linear Regression

In a multiple linear regression case with n features, our features are x1 through xn and our target is y. The algorithm finds the parameters that minimize the error between the actual values and the estimated values. The parameters the algorithm estimates are the coefficients of the features, b1 through bn, and the y-intercept, 
b0. ϵ is the error term or the residual value.

##### Polynomial Regression

In the case we have a polynomial function, we still have a linear model due to the fact that xi is in fact a feature and the coefficient/weight associated with that feature is still linear. To convert the original features into their higher order terms, we will use the PolynomialFeatures class provided by scikit-learn. Then, we train the model using Linear Regression.