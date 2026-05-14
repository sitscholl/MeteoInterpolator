from sklearn.linear_model import LinearRegression

from .base import BaseVerticalModel

class LinearVerticalModel(BaseVerticalModel):
    def __init__(self, **kwargs):
        self.estimator = LinearRegression(**kwargs)

    @classmethod
    def key(cls):
        return "linear"

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def _predict_numpy_2d(self, X):
        return self.estimator.predict(X)
