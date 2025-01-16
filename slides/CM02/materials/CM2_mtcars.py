import pandas as pd
import statsmodels.api as sm
import numpy as np

# load the dataset
filename = 'mtcars.csv'
df = pd.read_csv(filename, index_col=0)

# choose the predictors
X = df.drop(columns=['mpg'])
X['intercept'] = 1 # add columns of ones

# choose the observed variable
y = df['mpg']

# fit the multiple linear regression model
model = sm.OLS(y, X)
results = model.fit()

# print the summary of results
results.summary()


