from typing import cast, final

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.training.config import (
    PreprocessingConfig,
    TfidfConfig,
    TfidfPCAConfig,
    TransformerConfig,
)


arr_f32 = np.ndarray[None, np.dtype[np.float32]]


@final
class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        preprocessing_config: PreprocessingConfig,
    ) -> None:
        if isinstance(preprocessing_config, TfidfConfig):
            tfidf = TfidfVectorizer()
            self._pipeline = Pipeline([("tfidf", tfidf)])
        elif isinstance(preprocessing_config, TfidfPCAConfig):
            tfidf = TfidfVectorizer()
            pca = PCA(
                random_state=preprocessing_config.random_state,
                n_components=preprocessing_config.components,
            )
            self._pipeline = Pipeline([("tfidf", tfidf), ("pca", pca)])
        elif isinstance(preprocessing_config, TransformerConfig):
            self._transformer = SentenceTransformer(preprocessing_config.model_name)
        self.preprocessing_config = preprocessing_config

    def fit(self, X: list[str], y: arr_f32 | None = None) -> "Preprocessor":  # pyright: ignore[reportUnusedParameter]
        if isinstance(self.preprocessing_config, (TfidfConfig, TfidfPCAConfig)):
            _ = self._pipeline.fit(X)
        return self

    def transform(self, X: list[str]) -> arr_f32:
        if isinstance(self.preprocessing_config, (TfidfConfig, TfidfPCAConfig)):
            return cast(arr_f32, self._pipeline.transform(X))
        elif isinstance(self.preprocessing_config, TransformerConfig):
            return cast(arr_f32, self._transformer.encode(X, convert_to_numpy=True))
        else:
            raise NotImplementedError
