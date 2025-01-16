import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm, probplot

def simple_regression():

    data = fetch_california_housing(as_frame=True)
    df_X = data['data'][['MedInc']]
    df_X['intercept'] = np.ones(len(df_X))
    df_y = data['target']

    model = sm.OLS(df_y, df_X)
    results = model.fit()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_X['MedInc'], df_y, c='black', s=1)
    x_plot = np.linspace(0, 10, 100)
    beta_1 = results.params.iloc[1]
    beta_0 = results.params.iloc[0]
    y_plot = beta_1 * x_plot + beta_0
    ax.plot(x_plot, y_plot, c='C0', lw=3.0)

    return results, fig

def multiple_regression():

    data = fetch_california_housing(as_frame=True)
    df_X = data['data']
    df_X['intercept'] = np.ones(len(df_X))
    df_y = data['target']

    model = sm.OLS(df_y, df_X)
    results = model.fit()

    return results

data = fetch_california_housing(as_frame=True)
df_X = data['data'][['MedInc']]
df_X['intercept'] = np.ones(len(df_X))
df_y = data['target']

model = sm.OLS(df_y, df_X)
results = model.fit()

avg = results.params
std = results.bse

fig, ax = plt.subplots(figsize=(12, 5), ncols=2)
for i, axi in enumerate(ax):
    x = np.linspace(0.35, 0.55, 1000)
    y = norm.pdf(x, loc=avg.iloc[i], scale=std.iloc[i])
    ax[i].plot(x, y, lw=2.0, c='C0')
    ax[i].set_xticks([0.35, 0.40, 0.45, 0.50, 0.55])
    ax[i].set_title(avg.index[i])
fig.savefig('distribution_params.pdf', format='pdf')

residuals = results.resid

fig, ax = plt.subplots(figsize=(12, 5), ncols=2)
ax[0].scatter(df_X['MedInc'], residuals, c='k', s=1)
ax[0].set_ylim(-4.2, +4.2)
ax[0].axhline(y=0, lw=3.0, c='C0')
res = probplot(residuals, plot=ax[1])
fig.savefig('testing_residuals_01.pdf', format='pdf')

idx = df_y != df_y.max()
df_y_filter = np.log10(df_y[idx])
df_X_filter = df_X[idx]

model = sm.OLS(df_y_filter, df_X_filter)
results = model.fit()
residuals = results.resid

fig, ax = plt.subplots(figsize=(12.7, 4.2), ncols=3)
plt.subplots_adjust(wspace=0.20, left=0.05, right=0.95)
ax[0].scatter(df_X_filter['MedInc'], df_y_filter, c='black', s=1)
x_plot = np.linspace(0, 10, 100)
beta_1 = results.params.iloc[0]
beta_0 = results.params.iloc[1]
y_plot = beta_1 * x_plot + beta_0
ax[0].plot(x_plot, y_plot, c='C0', lw=3.0)
ax[0].set_title('log(Value) vs MedInc')
ax[1].scatter(df_X_filter['MedInc'], residuals, c='k', s=1)
ax[1].set_ylim(-4.2, +4.2)
ax[1].axhline(y=0, lw=3.0, c='C0')
ax[1].set_title('Residuals')
res = probplot(residuals, plot=ax[2])
ax[2].set_ylabel('')
fig.savefig('testing_residuals_02.pdf', format='pdf')