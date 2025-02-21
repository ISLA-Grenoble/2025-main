import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
N = 10_000
X = np.zeros((N, 2))
X[:, 0] = np.random.randn(N)
X[:, 1] = 3 * X[:, 0] + np.random.randn(N)
Y = X[:, 0] + X[:, 1] + 2 + np.random.randn(N)

df = pd.DataFrame()
df['Y'] = Y
df['intercept'] = np.ones(N)
df['X1'] = X[:, 0]
model_A = sm.OLS(df['Y'], df[['intercept', 'X1']])
results = model_A.fit()

print('Model A')
print(results.params)
print('sigma2_A = ', results.scale)
print('')

df = pd.DataFrame()
df['Y'] = Y
df['intercept'] = np.ones(N)
df['X2'] = X[:, 1]
model_B = sm.OLS(df['Y'], df[['intercept', 'X2']])
results = model_B.fit()

print('Model B')
print(results.params)
print('sigma2_B = ', results.scale)
print('')

N = 10
X = np.zeros((N, 2))
X[:, 0] = np.random.randn(N)
X[:, 1] = 3 * X[:, 0] + np.random.randn(N)
Y = X[:, 0] + X[:, 1] + 2 + np.random.randn(N)

df = pd.DataFrame()
df['Y'] = Y
df['intercept'] = np.ones(N)
df['X1'] = X[:, 0]
df['X2'] = X[:, 1]
model = sm.OLS(df['Y'], df[['intercept', 'X1', 'X2']])
results = model.fit()

print('full model')
print(results.params)
print('sigma2 = ', results.scale)