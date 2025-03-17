# pyright: reportAny = none

from dataclasses import asdict
from typing import final

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.training.config import (
    ClassifierConfig,
    LogisticRegressionConfig,
    RandomForestConfig,
    SVCConfig,
    classifiers,
)
from src.utils.types import arr_1d_f, arr_2d_f


@final
class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        config: ClassifierConfig,
    ) -> None:
        # Fix hydra's duck typing
        self.config = classifiers[config.name](**config)
        model_params = asdict(self.config)
        model_params.pop("name", None)
        if isinstance(self.config, LogisticRegressionConfig):
            self.model = LogisticRegression(**model_params)
        elif isinstance(self.config, SVCConfig):
            self.model = SVC(probability=True, **model_params)
        elif isinstance(self.config, RandomForestConfig):
            self.model = RandomForestClassifier(**model_params)
        else:
            raise NotImplementedError

    def fit(self, X: arr_2d_f, y: arr_1d_f) -> "Classifier":
        self.model_ = self.model.fit(X, y)  # pyright: ignore[reportUninitializedInstanceVariable]
        return self

    def predict(self, X: arr_2d_f) -> arr_1d_f:
        return self.model_.predict(X)

    def predict_proba(self, X: arr_2d_f) -> arr_1d_f:
        return self.model_.predict_proba(X)
