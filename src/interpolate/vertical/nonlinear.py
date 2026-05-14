
from .base import BaseVerticalModel

class NonLinearVerticalModel(BaseVerticalModel):
    def __init__(self):
        pass

    @classmethod
    def key(cls):
        return "non-linear"

    def fit(self, X, y):
        pass

    def _predict_numpy_2d(self, X):
        pass