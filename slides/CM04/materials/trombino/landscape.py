import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from numpy.random import rand
import numpy as np
import cv2

np.random.seed(42)

K = 15
C = 12 * rand(K, 2)
S = 1 + 4 * rand(K)

x1_array = np.linspace(-5, +15, 100)
x2_array = np.linspace(-5, +15, 100)
np.meshgrid(x1_array, x2_array)
X1, X2 = np.meshgrid(x1_array, x2_array)
X = np.stack([X1.flatten(), X2.flatten()]).T

pdf_list = []
for k in range(K):
    pdf_k = multivariate_normal(mean=C[k], cov=S[k]*np.eye(2)).pdf(X)
    pdf_list.append(pdf_k)
pdf = 1/K * np.sum(pdf_list, axis=0)

fig = plt.figure(figsize=(9.5, 9.5))
ax = fig.add_subplot(1, 1, 1, projection='3d')
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.05)
ax.plot(X1, X2, pdf.reshape(100, 100), lw=0.5, c='k')
ax.view_init(elev=24, azim=-37)
ax.grid(False)
ax.axis(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.set_zlim([0, 0.03])
fig.savefig('landscape_original.pdf', format='pdf', transparent=True)

gauss = multivariate_normal(mean=np.mean(C, axis=0), cov=6*np.eye(2)).pdf(X)
fig = plt.figure(figsize=(9.5, 9.5))
ax = fig.add_subplot(1, 1, 1, projection='3d')
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.05)
ax.plot(X1, X2, pdf.reshape(100, 100), lw=0.5, c='k', alpha=0.3)
ax.plot(X1, X2, gauss.reshape(100, 100), lw=0.5, c='b')
ax.view_init(elev=24, azim=-37)
ax.grid(False)
ax.axis(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_zlim([0, 0.03])
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
fig.savefig('landscape_gauss.pdf', format='pdf', transparent=True)

gauss = multivariate_normal(mean=np.mean(C, axis=0), cov=6*np.eye(2)).pdf(X)
fig = plt.figure(figsize=(9.5, 9.5))
ax = fig.add_subplot(1, 1, 1, projection='3d')
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.05)
ax.plot(X1, X2, gauss.reshape(100, 100), lw=0.5, c='k')
ax.view_init(elev=24, azim=-37)
ax.grid(False)
ax.axis(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_zlim([0, 0.03])
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
fig.savefig('landscape_gauss_2.pdf', format='pdf', transparent=True)

image_path = 'photos_face/aadroujm.jpg'
image = cv2.imread(image_path)
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
noise = np.random.randint(32, 127, size=(128, 128))
output_path = 'noise.jpg'
cv2.imwrite(output_path, noise)