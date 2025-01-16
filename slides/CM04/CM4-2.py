import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

# load the dataset
file_path = 'NAm2.txt'
df = pd.read_csv(file_path, delimiter=' ')
# select only the genetic markers as predictors
predictors = df.columns[8:]
# create the design matrix
X = df[predictors].values
# get the observed values to predict
y = df[['lat']].values

scaler = StandardScaler()
pca = PCA(n_components=4)
lr = LinearRegression()

cv = KFold(n_splits=5)
scores = []
for train_idx, test_idx in cv.split(X, y):
    # split the train and test paritions
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    # scale the data
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # reduce the dimensionality
    pca.fit(X_train_scaled)
    X_train_reduced = pca.transform(X_train_scaled)
    X_test_reduced = pca.transform(X_test_scaled)
    # fit linear regression and evaluate error
    lr.fit(X_train_reduced, y_train)
    scores.append(lr.score(X_test_reduced, y_test))
print(np.mean(scores))

est = make_pipeline(scaler, pca, lr)

cv = KFold(n_splits=5)
scores = []
for train_idx, test_idx in cv.split(X, y):
    # split the train and test paritions
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    # fit the pipeline and evaluate error
    est.fit(X_train, y_train)
    scores.append(est.score(X_test, y_test))
print(np.mean(scores))


