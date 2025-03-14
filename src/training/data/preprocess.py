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
    TransformerConfig,
)


arr_f32 = np.ndarray[None, np.dtype[np.float32]]


@final
class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        preprocessing_config: PreprocessingConfig,
    ) -> None:
        self.config = preprocessing_config
        if isinstance(self.config, TfidfConfig):
            tfidf = TfidfVectorizer()
            pipeline_elements: list[tuple[str, BaseEstimator]] = [("tfidf", tfidf)]
            if self.config.use_pca:
                pca = PCA(
                    random_state=self.config.random_state,
                    n_components=self.config.pca_config.n_components,
                )
                pipeline_elements.append(("pca", pca))
            self._pipeline = Pipeline(pipeline_elements)
        elif isinstance(self.config, TransformerConfig):
            self._transformer = SentenceTransformer(self.config.model_name)
        else:
            raise NotImplementedError

    def fit(self, X: list[str], y: arr_f32 | None = None) -> "Preprocessor":  # pyright: ignore[reportUnusedParameter]
        if isinstance(self.config, TfidfConfig):
            _ = self._pipeline.fit(X)
        return self

    def transform(self, X: list[str]) -> arr_f32:
        if isinstance(self.config, TfidfConfig):
            return cast(arr_f32, self._pipeline.transform(X))
        elif isinstance(self.config, TransformerConfig):
            return cast(arr_f32, self._transformer.encode(X, convert_to_numpy=True))
        else:
            raise NotImplementedError
