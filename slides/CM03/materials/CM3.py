import numpy as np
import pandas as pd
import statsmodels.api as sm
np.random.seed(0)
# number of variables
pt = 201
# number of predictors
p = pt - 1
# sample size
n = 30 * p
# generate data
D = np.random.randn(n, pt)
df = pd.DataFrame(data=D)
df = df.rename(columns={0:'Y'})
# do multiple linear regression
df['intercept'] = 1
model = sm.OLS(df['Y'], df.drop(columns='Y'))
results = model.fit()
print(results.summary())

#%%%

from sklearn.model_selection import KFold
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
kf = KFold(n_splits=5)
dataset = fetch_california_housing()
lm = LinearRegression()
X = dataset.data
y = dataset.target
scores = []
for idx_train, idx_test in kf.split(X):
    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]
    lm.fit(X_train, y_train)
    scores.append(lm.score(X_test, y_test))
print(np.mean(scores))

#%%

import pandas as pd
import matplotlib.pyplot as plt

company_list = ['COP', 'CVX', 'VLO', 'XOM']
quotes = pd.DataFrame()
for company in company_list:
    filename = f'financial_data/{company}.csv'
    this_quote = pd.read_csv(filename)
    quotes[company] = this_quote['open']
quotes.index = pd.to_datetime(this_quote['date'])
fig, ax = plt.subplots(figsize=(9.4, 5.7))
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.0)
quotes.plot(ax=ax)
fig.savefig('financial_quotes.pdf', format='pdf')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ShuffleSplit, cross_validate
X = quotes.drop(columns=["CVX"]) 
y = quotes["CVX"]
regressor = GradientBoostingRegressor()
cv = ShuffleSplit(n_splits=10)
results_cv = cross_validate(regressor, X, y, cv=cv)
print(f'Mean R2: {results_cv["test_score"].mean():.2f}')

from sklearn.model_selection import TimeSeriesSplit
X = quotes.drop(columns=["CVX"]) 
y = quotes["CVX"]
regressor = GradientBoostingRegressor()
cv = TimeSeriesSplit(n_splits=10)
results_cv = cross_validate(regressor, X, y, cv=cv)
print(f'Mean R2: {results_cv["test_score"].mean():.2f}')

#%%

import pandas as pd
import statsmodels.api as sm
import numpy as np

# load the dataset
filename = 'effectivelife.csv'
df = pd.read_csv(filename, index_col=0)
df['brand'] = df['brand'].astype("category")

# encode the categorical features
df_enc = pd.get_dummies(df, dtype=np.float64)
df_enc = df_enc.drop(columns=['brand_A'])

# choose the predictors
X = df_enc.drop(columns=['life'])
X['intercept'] = 1 # add columns of ones

# choose the observed variable
y = df_enc['life']

from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
cv = KFold(n_splits=4)
results_cv = cross_validate(
    regressor, X, y, cv=cv, scoring='neg_mean_squared_error')
print(f'MSE: {-results_cv["test_score"].mean():.2f}')

regressor = LinearRegression()
cv = ShuffleSplit(n_splits=4, random_state=1)
results_cv = cross_validate(
    regressor, X, y, cv=cv, scoring='neg_mean_squared_error')
print(f'MSE: {-results_cv["test_score"].mean():.2f}')


