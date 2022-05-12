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

