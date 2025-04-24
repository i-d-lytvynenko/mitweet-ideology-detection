# pyright: reportUnusedParameter = none, reportConstantRedefinition = none, reportExplicitAny = none, reportRedeclaration = none

import logging
import os
from collections.abc import Iterable
from typing import Any, cast, final

import numpy as np
import spacy
from joblib import dump, load
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.training.config import (
    PreprocessorConfig,
    TfidfConfig,
    TransformerConfig,
    preprocessors,
)
from src.utils.types import arr_1d_f, arr_2d_f


logging.getLogger("spacy").setLevel(logging.WARNING)


@final
class CachedTransformer:
    def __init__(self, cache_dir: str, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_filename = os.path.join(cache_dir, f"{model_name}.pkl")

        self.cache: dict[str, arr_1d_f]
        try:
            self.cache = load(self.cache_filename)
        except FileNotFoundError:
            self.cache = {}

    def encode(self, sentences: list[str]) -> arr_2d_f:
        sentence_dict: dict[str, arr_1d_f | None]
        sentence_dict = {sentence: None for sentence in sentences}
        unknown_sentences: list[str] = []
        for sentence in sentences:
            if sentence in self.cache:
                sentence_dict[sentence] = self.cache[sentence]
            else:
                unknown_sentences.append(sentence)

        if unknown_sentences:
            new_embeddings = cast(
                arr_2d_f, self.model.encode(unknown_sentences, convert_to_numpy=True)
            )
            embed_i = 0
            for sentence in sentence_dict:
                if sentence_dict[sentence] is not None:
                    continue
                self.cache[sentence] = new_embeddings[embed_i]
                sentence_dict[sentence] = new_embeddings[embed_i]
                embed_i += 1

            _ = dump(self.cache, self.cache_filename)
        return np.array(list(sentence_dict.values()))


@final
class Lemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        nlp = spacy.blank("en")
        _ = nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
        _ = nlp.initialize()
        self.nlp = nlp

    def fit(self, X: list[str], y: arr_1d_f | None = None) -> "Lemmatizer":
        return self

    def transform(self, X: list[str]) -> list[str]:
        lemmatized_text: list[str] = []
        for doc in self.nlp.pipe(X):
            lemmas = [
                token.lemma_
                for token in doc
                if not token.is_stop and not token.is_punct
            ]
            lemmatized_text.append(" ".join(lemmas))
        return lemmatized_text


@final
class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        config: PreprocessorConfig,
    ) -> None:
        # Fix hydra's duck typing
        self.config = preprocessors[config.name](**config)  # pyright: ignore[reportCallIssue]
        if isinstance(self.config, TfidfConfig):
            tfidf = TfidfVectorizer()
            pipeline_elements: list[tuple[str, BaseEstimator]] = [("tfidf", tfidf)]
            if self.config.lemmatize:
                lemmatizer = Lemmatizer()
                pipeline_elements.insert(0, ("lemmatizer", lemmatizer))
            if self.config.use_pca:
                pca = PCA(
                    random_state=self.config.random_state,
                    n_components=self.config.pca_config.n_components,
                )
                pipeline_elements.append(("pca", pca))
            self.pipeline = Pipeline(pipeline_elements)
        elif isinstance(self.config, TransformerConfig):
            self.transformer = CachedTransformer(
                cache_dir=self.config.cache_dir, model_name=self.config.model_name
            )
        else:
            raise NotImplementedError

    def fit(
        self,
        X: list[str] | np.ndarray[Any, Any] | Iterable[Any] | DataFrame,
        y: arr_1d_f | None = None,
    ) -> "Preprocessor":
        X: list[str] = list(X)
        if isinstance(self.config, TfidfConfig):
            self.pipeline_ = self.pipeline.fit(X)  # pyright: ignore[reportUninitializedInstanceVariable]
        return self

    def transform(
        self, X: list[str] | np.ndarray[Any, Any] | Iterable[Any] | DataFrame
    ) -> arr_2d_f:
        X: list[str] = list(X)
        if isinstance(self.config, TfidfConfig):
            return cast(arr_2d_f, self.pipeline_.transform(X))
        elif isinstance(self.config, TransformerConfig):
            return self.transformer.encode(X)
        else:
            raise NotImplementedError
