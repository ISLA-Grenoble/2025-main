import pandas as pd
import statsmodels.api as sm
import numpy as np

# load the dataset
filename = 'effectivelife.csv'
df = pd.read_csv(filename, index_col=0)
df['brand'] = df['brand'].astype("category")

# encode the categorical features
df_enc = pd.get_dummies(
    df, dtype=np.float64, drop_first=True)

# choose the predictors
X = df_enc.drop(columns=['life'])
X['intercept'] = 1 # add columns of ones

# choose the observed variable
y = df_enc['life']

# fit the multiple linear regression model
model = sm.OLS(y, X)
results = model.fit()

# print the summary of results
print(results.summary())




