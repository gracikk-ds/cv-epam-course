FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN pip install -U pip
RUN pip install poetry

ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID mlearning
RUN useradd -r --no-log-init --uid $USER_ID -g mlearning mlearning

# setting working dir
WORKDIR /training

# make dirs
RUN mkdir logs models data

# copy dependencies
COPY pyproject.toml /training/pyproject.toml

RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

# copy trainer module
COPY src /training/src

# setting working dir
WORKDIR /training/src

# USER mlearning

ENTRYPOINT ["python", "task.py"]
