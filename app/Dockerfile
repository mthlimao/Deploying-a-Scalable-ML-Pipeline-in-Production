FROM python:3.8-bookworm

ENV DEVENV_PATH=/udacity
ENV CONTAINER_PROJECT_PATH=${DEVENV_PATH}/ml_api
ENV CONTAINER_PROJECT_PATH_APP=${CONTAINER_PROJECT_PATH}/app
ENV PYTHONPATH "${PYTHONPATH}:${CONTAINER_PROJECT_PATH_APP}"

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        # deps for building python deps
        build-essential \
        iputils-ping && \
        apt clean

WORKDIR ${CONTAINER_PROJECT_PATH_APP}

COPY ./requirements.txt ./requirements.txt

RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt 

COPY ./main.py ./main.py
COPY ./sanitycheck.py ./sanitycheck.py
COPY ./setup.py ./setup.py

COPY ./data/ ./data/
COPY ./source/ ./source/
COPY ./model/ ./model/

EXPOSE 8000
ENTRYPOINT uvicorn main:app --host 0.0.0.0 --reload;