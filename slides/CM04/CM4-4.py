import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_validate

beta_0 = 3.0
beta_1 = 0.0
beta_2 = 4.0
beta = np.array([beta_0, beta_1, beta_2])

N_train = 1000
N_test = 1000

X_train = np.zeros((N_train, 3))
X_train[:, 0] = 8.0 * np.random.randn(N_train)
X_train[:, 1] = 3.0 * np.random.randn(N_train)
X_train[:, 2] = 0.1 * np.random.randn(N_train)
y_train = X_train @ beta + 0.10 * np.random.randn(N_train)

X_test = np.zeros((N_test, 3))
X_test[:, 0] = 0.1 * np.random.randn(N_test)
X_test[:, 1] = 3.0 * np.random.randn(N_test)
X_test[:, 2] = 7.0 * np.random.randn(N_test)
y_test = X_test @ beta + 0.10 * np.random.randn(N_test)

X = np.concatenate([X_train, X_test])
y = np.concatenate([y_train, y_test])

# reducing dimension before
pca = PCA(n_components=2)
X_red = pca.fit_transform(np.concatenate([X_train, X_test]))
X_red_train = X_red[:len(X_train)]
X_red_test = X_red[len(X_train):]
lr = LinearRegression()
lr.fit(X_red_train, y_train)
print(lr.score(X_red_test, y_test))

# reducing dimension at the right moment
pca = PCA(n_components=2)
X_red_train = pca.fit_transform(X_train)
X_red_test = pca.transform(X_test)
lr = LinearRegression()
lr.fit(X_red_train, y_train)
print(lr.score(X_red_test, y_test))


