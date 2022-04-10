FROM pytorch/pytorch

RUN pip install -U pip
RUN pip install poetry

# copy dataset
COPY dataset.tar ./dataset.tar
# extract it
RUN tar -xf dataset.tar && rm dataset.tar

# copy trainer module
COPY trainer_docker trainer_docker

# copy dependencies
COPY pyproject.toml ./pyproject.toml
RUN poetry install

# setting working dir
WORKDIR /trainer_docker

EXPOSE 8080

ENTRYPOINT ["python", "task.py"]
