from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier


def create(type, params):
    cls = globals().get(type, None)
    if cls is None:
        raise AttributeError(f'Unknown model type: "{type}"')

    return cls(**params)
