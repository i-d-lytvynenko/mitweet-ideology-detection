# pyright: reportAny=none, reportExplicitAny=none

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


cs = ConfigStore.instance()


@dataclass
class RandomState:
    random_state: int = 42


@dataclass
class PreprocessorConfig(RandomState):
    name: str = MISSING


@dataclass
class PCAConfig:
    n_components: int = 100


@dataclass
class TfidfConfig(PreprocessorConfig):
    name: str = "tfidf"
    use_pca: bool = False
    pca_config: PCAConfig = field(default_factory=PCAConfig)

    # Fix hydra's duck typing
    def __post_init__(self):
        if isinstance(self.pca_config, dict):
            self.pca_config = PCAConfig(**self.pca_config)


@dataclass
class TransformerConfig(PreprocessorConfig):
    name: str = "transformer"
    cache_dir: str = "embeddings_cache"
    model_name: str = "all-mpnet-base-v2"


preprocessors = {}
for preprocessor in [TfidfConfig, TransformerConfig]:
    cs.store(group="preprocessor", name=preprocessor.name, node=preprocessor)
    preprocessors[preprocessor.name] = preprocessor


@dataclass
class ClassifierConfig(RandomState):
    name: str = MISSING


@dataclass
class LogisticRegressionConfig(ClassifierConfig):
    name: str = "logreg"


@dataclass
class SVCConfig(ClassifierConfig):
    name: str = "svm"
    kernel: str = "rbf"


@dataclass
class RandomForestConfig(ClassifierConfig):
    name: str = "random_forest"
    n_estimators: int = 100


classifiers = {}
for classifier in [LogisticRegressionConfig, SVCConfig, RandomForestConfig]:
    cs.store(group="classifier", name=classifier.name, node=classifier)
    classifiers[classifier.name] = classifier


@dataclass
class BaseConfig(RandomState):
    data_path: str = "${hydra:runtime.cwd}/data/MITweet.csv"

    # Some facets don't have enough support
    facet_ids_to_skip: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 7])


@dataclass
class PaperComparisonConfig(BaseConfig):
    remove_irrelevant: bool = False
    importance_coefs: tuple[float, float, float, float] = (1, 1, 1, 1)


@dataclass
class Config(BaseConfig):
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"preprocessor": "tfidf"},
            {"classifier": "logreg"},
        ]
    )
    preprocessor: PreprocessorConfig = MISSING
    classifier: ClassifierConfig = MISSING


cs.store(name="paper_comparison_config", node=PaperComparisonConfig)
cs.store(name="config", node=Config)
