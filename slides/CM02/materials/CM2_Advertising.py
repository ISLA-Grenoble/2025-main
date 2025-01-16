import pandas as pd
import statsmodels.api as sm
import numpy as np

# load the dataset
filename = 'Advertising.csv'
df = pd.read_csv(filename, index_col=0)

# choose the predictors
X = df[['newspaper']]
X['intercept'] = 1 # add columns of ones

# choose the observed variable
y = df['sales']

# fit the multiple linear regression model
model = sm.OLS(y, X)
results = model.fit()

# print the summary of results
results.summary()


