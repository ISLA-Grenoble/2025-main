import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

GNP_capita = [2810, 1540, 24130, 7328, 2400, 620, 21890, 110, 25800, 21030, 14000, 1350]
Inflation = [14.7, 50.0, 3.5, 4.4, 440.8, 19.8, 4.2, 35.7, 4.1, 3.2, 3.3, 8.2]
Unemployement = [0.0, 24.3, 5.1, 0.0, 4.8, 17.5, 6.7, 0.0, 7.7, 9.4, 0.0, 15.0]
External_trade = [6.3, 4.2, 23.5, 22.1, 10.5, -5.9, -73.4, -0.6, 2.2, -10.1, 20.0, -0.9]
Population = [36.8, 26.1, 81.0, 14.8, 153.0, 56.1, 255.0, 54.6, 5.0, 57.2, 1.3, 8.6]
Area = [1.22, 2.38, 0.35, 2.15, 8.51, 1.00, 9.36, 1.22, 0.33, 0.55, 0.02, 0.16]
Country = ['South Africa', 'Algeria', 'Germany', 'Saudi Arabia', 'Brazil', 'Egypt', 'USA', 'Ethiopia', 'Finland', 'France', 'Kuwait', 'Tunisia']

df = pd.DataFrame()
df['GNB.capita'] = GNP_capita
df['inflation'] = Inflation
df['unemployement'] = Unemployement
df['external.trade'] = External_trade
df['population'] = Population
df['area'] = Area
df.index = Country
df.to_csv('economics.csv')

X = df.values
pca = PCA(n_components=2)
scaler = StandardScaler()
est = make_pipeline(scaler, pca)
est.fit(X)
Z = est.transform(X)
Q = est.steps[1][1].components_

fig, ax = plt.subplots(figsize=(6.6, 6.1))
for i in range(len(df)):
    ax.text(Z[i, 0], Z[i, 1], df.index[i])
ax.axvline(x=0, c='k', lw=0.8)
ax.axhline(y=0, c='k', lw=0.8)
ax.set_xlim(-2, 5)
ax.set_ylim(-3, 4)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
fig.savefig('CM4-pca-economics-scaled.pdf', format='pdf')

pca.fit(X)
Z = est.transform(X)
fig, ax = plt.subplots(figsize=(6.6, 6.1))
for i in range(len(df)):
    ax.text(Z[i, 0], Z[i, 1], df.index[i])
ax.axvline(x=0, c='k', lw=0.8)
ax.axhline(y=0, c='k', lw=0.8)
ax.set_xlim(-10255, -10245)
ax.set_ylim(-100, -94)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
fig.savefig('CM4-pca-economics-nonscaled.pdf', format='pdf')

fig, ax = plt.subplots(figsize=(6.6, 6.1))
circle = plt.Circle((0, 0), 1.0, color='k', fill=False)
ax.add_patch(circle)
for i, qi in enumerate(Q.T):
    ax.text(qi[0], qi[1], df.columns[i])
ax.axvline(x=0, c='k', lw=0.8)
ax.axhline(y=0, c='k', lw=0.8)
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.set_xticks([-1, -0.5, 0.0, +0.5, 1.0])
ax.set_yticks([-1, -0.5, 0.0, +0.5, 1.0])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
fig.savefig('CM4-pca-economics-Q-matrix.pdf', format='pdf')