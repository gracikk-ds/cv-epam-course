FROM pytorch/pytorch

RUN pip install -U pip

# copy dataset
COPY dataset.tar ./dataset.tar

# extract it
RUN tar -xf dataset.tar && rm dataset.tar

# copy trainer module
COPY trainer_docker trainer_docker

# copy setup file
COPY setup.py setup.py

# install module
RUN pip install -e .

# setting working dir
WORKDIR /trainer_docker

EXPOSE 8080

ENTRYPOINT ["python", "task.py"]
