import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def generate_parameters(epsilon):
    mu_0_a = np.array([0.0, -1.0])
    mu_0_b = np.array([0.0, +1.0])
    mu_1 = np.array([0.0, epsilon])
    sg_0 = np.eye(2) * 0.5
    sg_1 = np.eye(2) * 0.4
    return mu_0_a, mu_0_b, mu_1, sg_0, sg_1

def sample_class_0(epsilon):
    mu_0_a, mu_0_b, _, sg_0, _ = generate_parameters(epsilon)
    c = np.random.rand()
    if c < 0.5:
        x = multivariate_normal(mean=mu_0_a, cov=sg_0).rvs(size=1)
    else:
        x = multivariate_normal(mean=mu_0_b, cov=sg_0).rvs(size=1)
    return x

def pdf_class_0(x, epsilon):
    mu_0_a, mu_0_b, _, sg_0, _ = generate_parameters(epsilon)
    p_a = multivariate_normal(mean=mu_0_a, cov=sg_0).pdf(x)
    p_b = multivariate_normal(mean=mu_0_b, cov=sg_0).pdf(x)
    return 0.5 * p_a + 0.5 * p_b

def sample_class_1(epsilon):
    _, _, mu_1, _, sg_1 = generate_parameters(epsilon)
    x = multivariate_normal(mean=mu_1, cov=sg_1).rvs(size=1)
    return x

def pdf_class_1(x, epsilon):
    _, _, mu_1, _, sg_1 = generate_parameters(epsilon)
    p = multivariate_normal(mean=mu_1, cov=sg_1).pdf(x)
    return p

def generate_dataset(N, epsilon, p):
    y = (np.random.rand(N) < p).astype(int)
    x = []
    for yi in y:
        if yi == 0:
            xi = sample_class_0(epsilon)
        elif yi == 1:
            xi = sample_class_1(epsilon)
        x.append(xi)
    return np.stack(x), y

def make_figure(x, y):
    pass

eps = 2.0
p = 0.5
x_train, y_train = generate_dataset(N=1000, epsilon=eps, p=p)
x_test, y_test = generate_dataset(N=1000, epsilon=eps, p=p)
dataset_dict = {}
dataset_dict['train'] = (x_train, y_train)
dataset_dict['test'] = (x_test, y_test)

fig, ax = plt.subplots(figsize=(8.3, 7.0))
colors = {0:'C0', 1:'C1'}
markers = {'train':'o', 'test':'v'}
for dataset in ['train', 'test']:
    x, y = dataset_dict[dataset]
    for label in [0, 1]:
        select = y == label
        ax.scatter(x[select, 0], x[select, 1], s=30, c=colors[label], marker=markers[dataset], label=f'{dataset} : class {label}')

x1_array = np.linspace(-3, +3, 100)
x2_array = np.linspace(-3, +3, 100)
X1, X2 = np.meshgrid(x1_array, x2_array)
X = np.stack([X1.flatten(), X2.flatten()]).T
pdf_0 = pdf_class_0(X, epsilon=eps)
pdf_1 = pdf_class_1(X, epsilon=eps)
b = np.log(pdf_1) - np.log(pdf_0)
b = b.reshape(100, 100)
ax.contour(X1, X2, b, levels=[0], linestyles='dashed')
ax.plot([], [], c='k', label='Bayes', ls='--')

ax.legend()

# fig.show()

eps_array = np.linspace(-6, +6, 20)
acc = []
for eps in eps_array:
    x_test, y_test = generate_dataset(N=1000, epsilon=eps, p=p)
    y_pred = pdf_class_1(x_test, epsilon=eps) > pdf_class_0(x_test, epsilon=eps)
    acc.append(np.mean(y_test == y_pred))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(eps_array, acc, c='k', lw=2.0)
fig.show()
