# Metric Learning

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
