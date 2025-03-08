from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class RandomState:
    random_state: int = 42


@dataclass
class BaseConfig(RandomState):
    data_path: str = "${hydra:runtime.cwd}/data/MITweet.csv"

    # Some facets don't have enough support
    facet_ids_to_skip: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 7])


@dataclass
class PaperComparisonConfig(BaseConfig):
    remove_irrelevant: bool = False
    importance_coefs: tuple[float, float, float, float] = (1, 1, 1, 1)


cs = ConfigStore.instance()
cs.store(name="paper_comparison_config", node=PaperComparisonConfig)
