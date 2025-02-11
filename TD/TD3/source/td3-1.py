import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filename = './cars04.csv'
df = pd.read_csv(filename, index_col=0)
X = df.values[:, 7:]

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

scl = StandardScaler()
pca = PCA()
est = make_pipeline(scl, pca)

est.fit(X)

for i in range(X.shape[1]):
    variance = pca.explained_variance_[i]
    variance_ratio = pca.explained_variance_ratio_[i]
    print(f'PC{i+1:02d}:', f'{variance:.3f}', f'{variance_ratio:.3f}')

df_pc = pd.DataFrame()
df_pc['PC1'] = pca.components_[0, :]
df_pc['PC2'] = pca.components_[1, :]
df_pc.index = df.columns[7:]

# est.set_params(pca__n_components=2)
X_pca = est.fit_transform(X)

fig, axs = plt.subplots(figsize=(12.0, 5.5), ncols=2)

ax = axs[0]

std_y = pca.singular_values_ / np.sqrt(len(df))
var_x = np.ones(len(std_y))  # the data is being scaled

for i in range(X.shape[1]):
    point = df_pc.iloc[i].values
    if i == 0: # Retail
        ax.text(std_y[0]/var_x[0]*point[0] - 0.32,
                std_y[1]/var_x[1]*point[1],
                df_pc.index[i], color='k', size=14)        
    elif i == 1: # Dealer
        ax.text(std_y[0]/var_x[0]*point[0] - 0.30,
                std_y[1]/var_x[1]*point[1] - 0.12,
                df_pc.index[i], color='k', size=14) 
    elif i in [2, 3, 4, 7, 10]: # Engine, Cylinders, Horsepower, Weight, Width
        ax.text(std_y[0]/var_x[0]*point[0] - 0.35,
                std_y[1]/var_x[1]*point[1],
                df_pc.index[i], color='k', size=14)                
    elif i == 8: # Wheelbase
        ax.text(std_y[0]/var_x[0]*point[0] + 0.03,
                std_y[1]/var_x[1]*point[1] + 0.08,
                df_pc.index[i], color='k', size=14) 
    elif i == 9: # Length
        ax.text(std_y[0]/var_x[0]*point[0] + 0.06,
                std_y[1]/var_x[1]*point[1],
                df_pc.index[i], color='k', size=14) 
    elif i == 5: # CityMPG
        ax.text(std_y[0]/var_x[0]*point[0] - 0.25,
                std_y[1]/var_x[1]*point[1] + 0.10,
                df_pc.index[i], color='k', size=14)         
    elif i == 6: # HighwayMPG
        ax.text(std_y[0]/var_x[0]*point[0] - 0.50,
                std_y[1]/var_x[1]*point[1] - 0.15,
                df_pc.index[i], color='k', size=14)         
    else:
        ax.text(std_y[0]/var_x[0]*point[0],
                std_y[1]/var_x[1]*point[1],
                df_pc.index[i], color='k', size=14)                
    ax.scatter(std_y[0]/var_x[0]*point[0],
            std_y[1]/var_x[1]*point[1],
            s=120, edgecolor='k', facecolor='none')        
ax.axvline(x=0, c='k', lw=0.8)
ax.axhline(y=0, c='k', lw=0.8)
ax.set_xlim(-1, +1)
ax.set_ylim(-1, +1)
circle = plt.Circle((0, 0), 1, color='k', fill=False)
ax.add_patch(circle)

ax = axs[1]
ax.axvline(x=0, c='k', lw=0.8)
ax.axhline(y=0, c='k', lw=0.8)
for i, Xi in enumerate(X_pca):
    if i in [22, 122, 279]:
        pass
    else:
        ax.text(Xi[0], Xi[1], df.index[i], size=2)
for j in [22, 112, 279]:
    ax.text(X_pca[j, 0], X_pca[j, 1], df.index[j], size=12, c='r')
ax.set_xlim(-6, 12)
ax.set_ylim(-3.5, 4)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

fig.show()
