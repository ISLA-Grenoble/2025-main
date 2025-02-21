import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
from sklearn.decomposition import PCA

list_photos = sorted(glob('photos_face/*.jpg'))
L = 128
X = []
for filename in list_photos:
    img = cv2.imread(filename, 0).astype(np.float64)
    X.append(img.flatten())
X = np.stack(X)

est = PCA(n_components=64)
X_red = est.fit_transform(X)
eigenfaces = est.components_.reshape(-1, L, L)

fig, ax = plt.subplots(figsize=(13.5, 2.6), ncols=5)
plt.subplots_adjust(left=0.05, right=0.95)
for i, axi in enumerate(ax):
    axi.imshow(eigenfaces[i], cmap='gray_r')
    axi.set_xticks([])
    axi.set_yticks([])
    axi.set_title(f'PC{i+1} ({est.explained_variance_ratio_[i]:.2f} %)')
fig.savefig('eigenfaces.pdf', format='pdf')

i = 8
student = list_photos[i].split('/')[1].split('.')[0]
n_components = 4
img = cv2.imread(list_photos[i], 0).astype(np.float64)
img_red = np.sum([X_red[i, j] * eigenfaces[j] for j in range(n_components)], axis=0)
fig, ax = plt.subplots(figsize=(12, 6), ncols=2)
ax[0].imshow(img, cmap='gray')
ax[0].set_title('original photo')
ax[1].imshow(img_red, cmap='gray')
ax[1].set_title(f'reconstruction with {n_components} components')
for axi in ax:
    axi.set_xticks([])
    axi.set_yticks([])
fig.savefig(f'reconstruction_{student}_{n_components}_components.pdf', format='pdf')

i = 25
n_components_array = [4, 32, 64]
student = list_photos[i].split('/')[1].split('.')[0]
img = cv2.imread(list_photos[i], 0).astype(np.float64)
img_red = []
for n_components in n_components_array:
    img_red.append(np.sum([X_red[i, j] * eigenfaces[j] for j in range(n_components)], axis=0))
fig, ax = plt.subplots(figsize=(12, 6), ncols=4)
ax[0].imshow(img, cmap='gray')
ax[0].set_title('original photo')
for i, n_components in enumerate(n_components_array):
    ax[i+1].imshow(img_red[i], cmap='gray')
    ax[i+1].set_title(f'{n_components} components')
for axi in ax:
    axi.set_xticks([])
    axi.set_yticks([])
fig.savefig(f'reconstruction_{student}_multiple_components.pdf', format='pdf')

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(range(1, 64+1), est.explained_variance_, lw=2.0, c='k')
ax.scatter(range(1, 64+1), est.explained_variance_, s=20, c='k')
ax.set_xlabel('component', fontsize=14)
ax.set_ylabel('explained variance', fontsize=14)
fig.savefig(f'explainedvariance_curve.pdf', format='pdf', transparent=False)