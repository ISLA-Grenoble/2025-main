import numpy as np
from sklearn.linear_model import LinearRegression, QuantileRegressor, HuberRegressor
import matplotlib.pyplot as plt

rng = np.random.RandomState(42)
x = np.linspace(start=0, stop=10, num=30)

X = x[:, np.newaxis]
y_true_mean = 10 + 0.5 * x
y = y_true_mean + rng.normal(loc=0, scale=0.5, size=x.shape[0])

X_outliers = rng.normal(0, 0.5, size=(4, 1))
y_outliers = rng.normal(0, 2.0, size=4)
X_outliers[:2, :] += X.max() + X.mean() / 2.0
X_outliers[2:, :] += X.min() - X.mean() / 2.0
y_outliers[:2] += y.min() - y.mean() / 2.0
y_outliers[2:] += y.max() + y.mean() / 2.0

X = np.vstack((X, X_outliers))
y = np.concatenate((y, y_outliers))

y_true_mean_plot = 10 + 0.5 * X[:, 0]

model = LinearRegression()
model.fit(X, y)
y_pred_L2 = model.predict(X)

model = QuantileRegressor(quantile=0.5, alpha=0.0)
model.fit(X, y)
y_pred_L1 = model.predict(X)

fig, ax = plt.subplots(figsize=(6.1, 6.0))
ax.scatter(X[:,0], y, color='b', alpha=0.5)
ax.plot(X[:,0], y_true_mean_plot, ls='--', color='black')
ax.plot(X[:,0], y_pred_L1, label='L1', color='green')
ax.plot(X[:,0], y_pred_L2, label='L2', color='red')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()

fig.savefig('plot.pdf', format='pdf')