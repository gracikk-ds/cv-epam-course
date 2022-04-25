FROM pytorch/pytorch

RUN pip install -U pip
RUN pip install poetry

RUN mkdir /training

# copy dataset
COPY dataset.tar /training/dataset.tar
# extract it
RUN tar -xf /training/dataset.tar && rm /training/dataset.tar

# copy trainer module
COPY src /training/src

# copy dependencies
COPY pyproject.toml /training/pyproject.toml

# setting working dir
WORKDIR /training

RUN poetry config virtualenvs.create false
RUN poetry install

# setting working dir
WORKDIR /training/src

ENTRYPOINT ["python", "task.py"]
