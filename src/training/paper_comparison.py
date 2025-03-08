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
from src.training.dataset_info import facets, paper_results


@hydra.main(version_base=None, config_name="paper_comparison_config")
def main(cfg: PaperComparisonConfig) -> None:
    df = pd.read_csv(cfg.data_path)

    RANDOM_STATE = cfg.random_state
    REMOVE_IRRELEVANT = cfg.remove_irrelevant
    IMPORTANCE_COEFS = np.array(cfg.importance_coefs)
    FACET_IDS_TO_SKIP = cfg.facet_ids_to_skip

    undersampler = RandomUnderSampler(random_state=RANDOM_STATE)
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("logreg", LogisticRegression(random_state=RANDOM_STATE)),
        ]
    )
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print(
        "Facet name".ljust(30),
        "Support".ljust(10) if REMOVE_IRRELEVANT else "",
        "F1".ljust(20),
        "F1 Reference" if REMOVE_IRRELEVANT else "",
        sep="",
    )

    for facet_id, facet_name in enumerate(facets, start=1):
        if facet_id in FACET_IDS_TO_SKIP:
            continue
        texts = df.loc[:, "tweet"].to_numpy()
        labels = df.loc[:, f"I{facet_id}"].to_numpy()
        if REMOVE_IRRELEVANT:
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

            y_pred: np.ndarray[None, np.dtype[np.float32]]

            if REMOVE_IRRELEVANT:
                y_pred = np.argmax(IMPORTANCE_COEFS[1:] * y_pred_proba, axis=1)
                scores.append(f1_score(y_test, y_pred, average="macro"))
            else:
                y_pred = np.argmax(IMPORTANCE_COEFS * y_pred_proba, axis=1) - 1
                # Don't use irrelevant tweets in F1 scoring
                scores.append(np.mean(f1_score(y_test, y_pred, average=None)[1:]))  # pyright: ignore[reportIndexIssue]

        f1s_mean = np.mean(scores) * 100
        f1s_std = np.std(scores) * 100
        print(
            facet_name.ljust(30),
            str(texts.size).ljust(10) if REMOVE_IRRELEVANT else "",
            f"{f1s_mean:.2f} (STD {f1s_std:.2f})".ljust(20),
            f"{paper_results[facet_id - 1]}" if REMOVE_IRRELEVANT else "",
            sep="",
        )


if __name__ == "__main__":
    main()
