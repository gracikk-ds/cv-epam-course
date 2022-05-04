FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN pip install -U pip
RUN pip install poetry

RUN mkdir /training

# copy processed data
COPY data/processed/dataset.tar /training/data/processed/dataset.tar

# extract it
RUN mkdir /training/data/processed/dataset
RUN tar -xf /training/data/processed/dataset.tar -C /training/data/processed/dataset --strip-components 1
RUN rm /training/data/processed/dataset.tar

# copy model folder
COPY models /training/models

# copy logs folder
COPY logs /training/logs

# copy trainer module
COPY src /training/src

# copy dependencies
COPY pyproject.toml /training/pyproject.toml
COPY poetry.lock /training/poetry.lock

# setting working dir
WORKDIR /training

RUN poetry config virtualenvs.create false
RUN poetry install

# setting working dir
WORKDIR /training/src

ENTRYPOINT ["python", "task.py"]
