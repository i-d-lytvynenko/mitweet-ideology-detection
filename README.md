# Multifaceted Ideology Detection on MITweet using Hydra, MLFlow, and Sentence Transformers

This project demonstrates a pipeline for detecting multifaceted ideology in tweets from the MITweet dataset using Hydra for configuration management, MLflow for experiment tracking and reproducibility, and Sentence Transformers for text embedding and classification with scikit-learn models.

The [MITweet](https://github.com/LST1836/MITweet) dataset is used.


## Setup

1. **Install dependencies:** `poetry install`
2. **Download MITweet dataset:**  Place the [MITweet.csv](https://github.com/LST1836/MITweet/raw/refs/heads/main/data/MITweet.csv) in the `data/` directory.
3. **Configure Hydra:** Adjust the parameters in `src/training/config.py` as needed.

Default config (run `python src/training/train.py --help` to see your config and more info about hydra usage):
```yaml
random_state: 42
data_path: ${hydra:runtime.cwd}/data/MITweet.csv
facet_ids_to_skip:
- 1
- 2
- 3
- 4
- 6
- 7
preprocessor:
  random_state: 42
  name: tfidf
  use_pca: false
  pca_config:
    n_components: 100
classifier:
  random_state: 42
  name: logreg
```

All configuration options can be found in `src/training/config.py`.

NOTE: `facet_ids_to_skip` is a list of facet ids (from 1 to 12), which don't have enough support in the dataset (< 1000 relevant samples).


## Running the experiment

To start the training, run
```
python src/training/train.py
```

To override config defaults, run something along
```
python src/training/train.py classifier=svm classifier.kernel=linear
```

To try multiple config options, run something along
```
python src/training/train.py classifier=svm classifier.kernel=linear,rbf --multirun`
```


## Running the baseline

Script `paper_comparison.py` compares a logistic regression on top of TF-IDF with the best results for each facet obtained in the [original paper](https://aclanthology.org/2023.emnlp-main.256). However, note that the authors filtered out irrelevant tweets before classification while this project doesn't. This script supports both options: change the value of `remove_irrelevant`.

It is also possible to shift classifier probability thresholds by changing values of `importance_coefs`.

Default config:
```yaml
random_state: 42
data_path: ${hydra:runtime.cwd}/data/MITweet.csv
facet_ids_to_skip:
- 1
- 2
- 3
- 4
- 6
- 7
remove_irrelevant: false
importance_coefs:
- 1.0
- 1.0
- 1.0
- 1.0
```

To start the training, run:
```
python src/training/paper_comparison.py remove_irrelevant=true
```


## MLflow tracking

Experiment runs, metrics, and models are logged automatically. You can access the MLflow UI (if MLflow server is running) to monitor progress and compare experiments.

To start MLFlow server, run `mlflow ui`.
