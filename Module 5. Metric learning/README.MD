# Metric Learning

###  Main things to know:
- [Lecture 1](https://www.youtube.com/watch?v=aU9yEwgrJ54) - highly recommended
- [Lecture 2](https://www.youtube.com/watch?v=bvcC4hXaIJY)
- [Lecture 3](https://www.youtube.com/watch?v=mr9njs6dess) - highly recommended
- [Pytorch metric learning framework](https://kevinmusgrave.github.io/pytorch-metric-learning/)

## Task
Build a system which can find a visually similar product in the catalog by a photo-query
We could separate the task into 2 main parts:
- The first one is two train the model using metric learning approach
- The second one is to develop demo stand where we could upload a photo and get response from the model with visually similar product

## Dataset
To train the model we used The Stanford Online Products dataset. You could download it [here](ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip)

## Project structure
```log
├── data/
│   ├── interim/ <- Intermediate data that has been transformed.
│   │   ├── dataset_full/
│   │   │   ├── test/
│   │   │   └── train/
│   │   └── dataset_part/
│   │       ├── test/
│   │       └── train/
│   ├── processed/ <- The final, canonical data sets for modeling.
│   │   └── dataset.tar
│   └── raw/ <- The original, immutable data dump.
│       └── Stanford_Online_Products/
├── demo/  <- code to run the demo stand
│   ├── __init__.py
│   ├── branded/
│   │   └── flower-logo.png
│   ├── docker-compose.yml
│   ├── inference.py
│   ├── main.py
│   ├── model.py
│   ├── pickles/
│   │   ├── data_pathes.csv
│   │   ├── embeddings.pickle
│   │   ├── mapper_faces.pickle
│   │   └── normalizer.pickle
│   ├── README.md  <- check this file to know how to run the demo
│   ├── settings.json
│   └── visualization.py
├── Dockerfile
├── logs/  <- training artifacts storage
├── models/  <- Trained and serialized models, model predictions, or model summaries
│   └── torchscript.pt
├── notebooks/  <- Jupyter notebooks
│   └── embedder_training.ipynb
├── poetry.lock
├── pyproject.toml
├── README.MD
├── reports/
│   └── figures/  <- Dataset structure exploratory analysis
│       ├── class_distribution.png
│       └── per_class_visualization.png
├── scripts/  <- Scripts to download or generate data
│   ├── dataset_statistics.py
│   └── make_dataset.py
├── src/  <- source code to train the model
│   ├── __init__.py
│   ├── metric_learning.py
│   ├── task.py
│   └── transformations.py
└── structure.py
```

## Run docker to train the model
To up docker container run the following command inside current directory:

```bash
docker build . -t grac20101/aleksandr-gordeev:metric-learning \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=123
```

```bash
docker run -d --gpus all --shm-size=16GB \
    --name mlearning \
    --mount type=bind,source="$(pwd)"/logs,target=/training/logs \
    --mount type=bind,source="$(pwd)"/models,target=/training/models \
    --mount type=bind,source="$(pwd)"/data,target=/training/data \
  grac20101/aleksandr-gordeev:metric-learning
```

## Conda + Poetry usage
```bash
# install poetry macos
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

# activate conda env and install dependencies
conda activate your_env
poetry config virtualenvs.path "path/to/your/conda/envs"
poetry config virtualenvs.create false
poetry install
```

## Pre-commit hooks
```bash
pip install pre-commit
pip install black
pre-commit install --install-hooks
```
