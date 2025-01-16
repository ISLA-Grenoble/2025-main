import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

df = pd.read_csv('food.csv', index_col=0)

pca = PCA(n_components=2)
scaler = StandardScaler()

X = df.values
Z0 = pca.fit_transform(X)

X = df.values
est = make_pipeline(scaler, pca)
Z1 = est.fit_transform(X)

X = df.values
rows = np.sum(X, axis=1)
X_ = X / rows[:, None]
est = make_pipeline(scaler, pca)
Z2 = est.fit_transform(X_)

fig, ax = plt.subplots(figsize=(13.5, 4.0), ncols=3)
plt.subplots_adjust(wspace=0.15)
for i in range(len(df)):
    ax[0].text(Z0[i, 0], Z0[i, 1], df.index[i])
ax[0].set_xlim(-500, 5500)
ax[0].set_ylim(-300, 1300)
for i in range(len(df)):
    ax[1].text(Z1[i, 0], Z1[i, 1], df.index[i])
ax[1].set_xlim(-4, 26)
ax[1].set_ylim(-4, 10)
for i in range(len(df)):
    ax[2].text(Z2[i, 0], Z2[i, 1], df.index[i])
ax[2].set_xlim(-6, 7)
ax[2].set_ylim(-6, 5)
# fig.savefig('CM4-food.pdf', format='pdf')

X = df.values
rows = np.sum(X, axis=1)
X_ = X / rows[:, None]
est = make_pipeline(scaler, pca)
Z2 = est.fit_transform(X_)
Q = est.steps[1][1].components_.T

fig, ax = plt.subplots(figsize=(7.2, 6.8))
plt.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.98)
for i in range(len(df)):
    ax.text(Z2[i, 0], Z2[i, 1], df.index[i])
for j in range(len(Q)):
    ax.arrow(x=0.0, y=0.0, dx=15*Q[j][0], dy=15*Q[j][1],
             ec='r', fc='r', lw=0.8, head_width = 0.1)
    ax.text(x=15*Q[j][0], y=15*Q[j][1], s=df.columns[j], c='r')
ax.set_xlim(-6, 7)
ax.set_ylim(-6, 6)
fig.savefig('CM4-food-biplot.pdf', format='pdf')

