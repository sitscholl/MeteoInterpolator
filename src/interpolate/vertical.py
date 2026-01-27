from sklearn.linear_model import LinearRegression

class VerticalModel:
    def __init__(self, **kwargs):
        self._params = dict(kwargs)
        self._model = LinearRegression(**kwargs)

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def get_params(self):
        return dict(self._params)

    def set_params(self, **kwargs):
        self._params.update(kwargs)
        self._model.set_params(**kwargs)
        return self