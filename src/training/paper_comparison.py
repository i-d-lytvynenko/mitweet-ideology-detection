# pyright: reportUnknownVariableType = none, reportUnknownArgumentType = none

import hydra
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.training.config import PaperComparisonConfig
from src.training.data.dataset_info import facets, paper_results
from src.utils.types import arr_1d_f


@hydra.main(version_base=None, config_name="paper_comparison_config")
def main(cfg: PaperComparisonConfig) -> None:
    df = pd.read_csv(cfg.data_path)

    undersampler = RandomUnderSampler(random_state=cfg.random_state)
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("logreg", LogisticRegression(random_state=cfg.random_state)),
        ]
    )
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)

    print(
        "Facet name".ljust(30),
        "Support".ljust(10) if cfg.remove_irrelevant else "",
        "F1".ljust(20),
        "F1 Reference" if cfg.remove_irrelevant else "",
        sep="",
    )

    for facet_id, facet_name in enumerate(facets, start=1):
        if facet_id in cfg.facet_ids_to_skip:
            continue
        texts = df.loc[:, "tweet"].to_numpy()
        labels = df.loc[:, f"I{facet_id}"].to_numpy()
        if cfg.remove_irrelevant:
            texts = texts[labels != -1]
            labels = labels[labels != -1]

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
            if cfg.remove_irrelevant:
                y_pred = np.argmax(cfg.importance_coefs[1:] * y_pred_proba, axis=1)
                scores.append(f1_score(y_test, y_pred, average="macro"))
            else:
                y_pred = np.argmax(cfg.importance_coefs * y_pred_proba, axis=1) - 1
                # Don't use irrelevant tweets in F1 scoring
                scores.append(np.mean(f1_score(y_test, y_pred, average=None)[1:]))  # pyright: ignore[reportIndexIssue]

        f1_mean = np.mean(scores) * 100
        f1_std = np.std(scores) * 100
        print(
            facet_name.ljust(30),
            str(texts.size).ljust(10) if cfg.remove_irrelevant else "",
            f"{f1_mean:.2f} (STD {f1_std:.2f})".ljust(20),
            f"{paper_results[facet_id - 1]}" if cfg.remove_irrelevant else "",
            sep="",
        )


if __name__ == "__main__":
    main()
