import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def make_data(n):
    x = 2 * np.random.rand(n) - 1
    y = x**3 + 2*np.exp(-6 * (x - 0.3)**2)
    y = y + 0.2 * np.abs(np.sin(2 * x) + 2) * np.random.randn(n)
    return x, y

n = 1000
x, y =  make_data(n)

fig, ax = plt.subplots(figsize=(6.6, 5.9))
ax.scatter(x, y, s=10, c='gray', alpha=0.3)
xx = np.linspace(-1, +1, 100)
yy = xx**3 + 2*np.exp(-6 * (xx - 0.3)**2)
ax.set_xlabel(r'$X$', fontsize=14)
ax.set_ylabel(r'$Y$', fontsize=14)
ax.set_xticks([-1, -0.5, 0, +0.5, 1])
fig.savefig('figure_curves_01.pdf', format='pdf')

fig, ax = plt.subplots(figsize=(6.6, 5.9))
ax.scatter(x, y, s=10, c='gray', alpha=0.3)
xx = np.linspace(-1, +1, 100)
yy = xx**3 + 2*np.exp(-6 * (xx - 0.3)**2)
ax.plot(xx, yy, c='C0', lw=3.0)
ax.set_xlabel(r'$X$', fontsize=14)
ax.set_ylabel(r'$Y$', fontsize=14)
ax.set_xticks([-1, -0.5, 0, +0.5, 1])
fig.savefig('figure_curves_02.pdf', format='pdf')

fig, ax = plt.subplots(figsize=(6.6, 5.9))
ax.scatter(x, y, s=10, c='gray', alpha=0.3)
xx = np.linspace(-1, +1, 100)
yy = xx**3 + 2*np.exp(-6 * (xx - 0.3)**2)
ax.plot(xx, yy, c='C0', lw=3.0)
ax.plot(xx, yy + 2 * 0.2 * np.abs(np.sin(2 * xx) + 2), c='r', ls='--')
ax.plot(xx, yy - 2 * 0.2 * np.abs(np.sin(2 * xx) + 2), c='r', ls='--')
ax.set_xlabel(r'$X$', fontsize=14)
ax.set_ylabel(r'$Y$', fontsize=14)
ax.set_xticks([-1, -0.5, 0, +0.5, 1])
fig.savefig('figure_curves_03.pdf', format='pdf')