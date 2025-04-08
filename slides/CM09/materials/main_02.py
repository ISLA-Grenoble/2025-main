import numpy as np
from sklearn.linear_model import LinearRegression, QuantileRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from sklearn.datasets import fetch_california_housing
california_housing = fetch_california_housing(as_frame=True)

X = california_housing.data[::50]
y = california_housing.target[::50]
idx = y < 5.0
X = X[idx]
y = y[idx]

bin, edges = np.histogram(y, bins=50)
edges = edges[:-1] + np.diff(edges) / 2

fig, ax = plt.subplots(figsize=(6.1, 6.0))
ax.plot(edges, bin, color='black')
ax.set_xlabel('house prices')
ax.set_ylabel('count')
ax.set_title('California housing prices')
ax.set_xlim(0, 5.0)
# ax.set_ylim(0, 600)
# ax.set_yticks(np.arange(0, 0.6, 0.1))
ax.set_xticks(np.arange(0, 6, 1))
# fig.show()

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print(mean_squared_error(y, y_pred))
print(mean_absolute_percentage_error(y, y_pred))

model = QuantileRegressor(quantile=0.5, alpha=0.0)
model.fit(X, y, sample_weight=1.0 / np.abs(y))
y_pred = model.predict(X)

print(mean_squared_error(y, y_pred))
print(mean_absolute_percentage_error(y, y_pred))