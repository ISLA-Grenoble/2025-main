import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

N = 1000

eps1 = np.random.randn(N)
eps2 = np.random.randn(N)
eps3 = np.random.randn(N)

X1 = eps1
X2 = 3*X1 + eps2

fig, ax = plt.subplots(figsize=(4.8, 4.7))
ax.scatter(X1, X2, s=10, c='k')
ax.set_xlim(-5, +5)
ax.set_ylim(-10, +10)
fig.savefig('figure_elipse.pdf', format='pdf')
Y = X2 + X1 + 2 + eps3