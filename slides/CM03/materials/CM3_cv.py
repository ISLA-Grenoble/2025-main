import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit

np.random.seed(6)

# generate dataset
N = 100
x = np.linspace(-5, +5, N)
y = x - x**2 + x**3 + 20 * np.random.randn(N)
X = x.reshape(-1, 1)

# instantiate pipeline
poly = PolynomialFeatures(include_bias=False)
lr = LinearRegression()
pipe = make_pipeline(poly, lr)

# instantiate grid search cv
degrees_array = np.arange(1, 10+1)
parameters = {'polynomialfeatures__degree':degrees_array}
cv = ShuffleSplit(n_splits=10)
est = GridSearchCV(estimator=pipe, param_grid=parameters, cv=cv, 
                   scoring='neg_mean_squared_error', return_train_score=True)

# get cv results
est.fit(X, y)
cv_test_scores = -est.cv_results_['mean_test_score']
cv_train_scores = -est.cv_results_['mean_train_score']

# plot results
fig, ax = plt.subplots(figsize=(6, 5.4))
ax.plot(degrees_array, cv_test_scores, c='C0', lw=3, label='test')
ax.plot(degrees_array, cv_train_scores, c='C1', lw=3, label='train')
ax.set_xlabel('degree')
ax.set_ylabel('MSE')
ax.set_xticks(degrees_array)
ax.legend()
fig.savefig('CM3_cv_figure.pdf', format='pdf')


