# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

rng = np.random.RandomState(1)
X = np.linspace(0, 6, 100)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

regr_1 = DecisionTreeRegressor(max_depth=4)

regr_2 = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=rng
)

regr_1.fit(X, y)
regr_2.fit(X, y)

y_1 = regr_1.predict(X)
y_2 = regr_2.predict(X)

import matplotlib.pyplot as plt
import seaborn as sns

colors = sns.color_palette("colorblind")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X, y, color=colors[0], label="training samples")
ax.plot(X, y_1, color=colors[1], label="n_estimators=1", linewidth=2)
ax.set_xlabel("data")
ax.set_ylabel("target")
ax.legend()
fig.show()
fig.savefig("example-regressor-1.pdf", format="pdf")