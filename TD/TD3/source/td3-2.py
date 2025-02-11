import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

df = pd.read_csv('foodFrance.csv', index_col=0)
X = df.values[:, 2:].astype(np.float64)
acronyms = ['BC2', 'WC2', 'UC2', 'BC3', 'WC3', 'UC3', 'BC4', 'WC4', 'UC4', 'BC5', 'WC5', 'UC5']

pca = PCA(n_components=2)
Xpca = pca.fit_transform(X)

df_pc = pd.DataFrame()
df_pc['PC1'] = pca.components_[0, :]
df_pc['PC2'] = pca.components_[1, :]
df_pc.index = df.columns[2:]

fig, axs = plt.subplots(figsize=(12.0, 5.5), ncols=2)

ax = axs[0]
circle = plt.Circle((0, 0), 1, color='C0', lw=2.0, fill=False)
ax.add_patch(circle)
std_y = pca.singular_values_ / np.sqrt(len(df))
std_x = np.std(X, axis=0)
points = df_pc.multiply(std_y, axis=1).divide(std_x, axis=0).values
for i, pi in enumerate(points):
    # ax.scatter(pi[0], pi[1], s=120, edgecolor='k', facecolor='none')
    ax.scatter(pi[0], pi[1], s=120, color='C3')
ax.text(points[0][0] - 0.35, points[0][1] - 0.01, 'Bread', size=14)   
ax.text(points[1][0] - 0.60, points[1][1], 'Vegetables', size=14)   
ax.text(points[2][0] - 0.32, points[2][1], 'Fruits', size=14)   
ax.text(points[3][0] - 0.30, points[3][1] + 0.07, 'Meat', size=14)   
ax.text(points[4][0] - 0.35, points[4][1] - 0.10, 'Poultry', size=14)   
ax.text(points[5][0] - 0.02, points[5][1] - 0.14, 'Milk', size=14)   
ax.text(points[6][0] + 0.05, points[6][1] + 0.03, 'Wine', size=14)   
ax.axvline(x=0, c='k', lw=0.8)
ax.axhline(y=0, c='k', lw=0.8)
ax.set_xlim(-1.05, +1.05)
ax.set_ylim(-1.05, +1.05)
ax.set_title('Variable space')

ax = axs[1]
colors = {'Blue collar': 'b', 'White collar': 'g', 'Upper class': 'r'}
for socialclass in colors.keys():
    idx = (df['Class'] == socialclass)
    ax.scatter(Xpca[idx, 0], Xpca[idx, 1], s=100, c=colors[socialclass])
for i, Xi in enumerate(Xpca):
    ax.text(Xi[0] + 15, Xi[1] + 10, s=acronyms[i], size=14)
ax.axvline(x=0, c='k', lw=0.8)
ax.axhline(y=0, c='k', lw=0.8)
ax.set_xlim(-720, 1250)
ax.set_ylim(-250, 320)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Data space')
fig.show()