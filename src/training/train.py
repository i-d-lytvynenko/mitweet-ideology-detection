# pyright: reportUnknownVariableType = none, reportUnknownArgumentType = none
from typing import Any

import hydra
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from mlflow.models import infer_signature
from omegaconf import DictConfig
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.training.config import Config
from src.training.data.dataset_info import facet_abbrevs
from src.training.data.preprocess import Preprocessor
from src.training.model import Classifier
from src.utils.types import arr_1d_f


def log_params(cfg: dict[str, Any] | DictConfig, parent_key: str = ""):  # pyright: ignore[reportExplicitAny]
    for key, value in cfg.items():
        new_key = f"{parent_key}.{key}" if parent_key else str(key)
        if isinstance(value, (dict, DictConfig)):
            log_params(value, new_key)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                log_params(item, f"{new_key}[{i}]")
        else:
            mlflow.log_param(new_key, value)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    with mlflow.start_run():
        log_params(cfg)  # pyright: ignore[reportArgumentType]
        df = pd.read_csv(cfg.data_path)

        # For some reason there is a tweet with 2 different labels
        df.drop_duplicates(subset=["tweet"], inplace=True)

        undersampler = RandomUnderSampler(random_state=cfg.random_state)
        preprocessor = Preprocessor(cfg.preprocessor)
        classifier = Classifier(cfg.classifier)
        pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", classifier)]
        )
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)

        filtered_facet_abbrevs = [
            facet_abbrevs[i]
            for i in range(len(facet_abbrevs))
            if i + 1 not in cfg.facet_ids_to_skip
        ]
        for facet_id, facet_abbrev in tqdm(
            enumerate(filtered_facet_abbrevs, start=1),
            total=len(filtered_facet_abbrevs),
            desc="Processing facets",
        ):
            texts = df.loc[:, "tweet"].to_numpy()
            labels = df.loc[:, f"I{facet_id}"].to_numpy()

            scores = []
            for train_index, test_index in kf.split(texts, labels):
                X_train, X_test = texts[train_index], texts[test_index]
                y_train, y_test = labels[train_index], labels[test_index]

                X_train, y_train = undersampler.fit_resample(  # pyright: ignore[reportAssignmentType]
                    X_train.reshape((-1, 1)), y_train
                )
                X_train = X_train[:, 0]
                y_pred_proba = pipeline.fit(X_train, y_train).predict_proba(X_test)

                y_pred: arr_1d_f
                y_pred = np.argmax(y_pred_proba, axis=1) - 1
                # Don't use irrelevant tweets in F1 scoring
                scores.append(np.mean(f1_score(y_test, y_pred, average=None)[1:]))  # pyright: ignore[reportIndexIssue]

            f1_mean = round(float(np.mean(scores)), 4)
            f1_std = round(float(np.std(scores)), 4)
            _ = mlflow.log_metric(f"f1_{facet_abbrev}", f1_mean)
            _ = mlflow.log_metric(f"std_f1_{facet_abbrev}", f1_std)
            signature = infer_signature(X_test, y_pred)  # pyright: ignore[reportPossiblyUnboundVariable]
            _ = mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path=cfg.trained_models_path,
                signature=signature,
            )


if __name__ == "__main__":
    main()
