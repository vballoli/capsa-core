
# Dockerfile for the installed package

FROM ubuntu:latest

ARG PY_VERSION
ENV PY_VERSION ${PY_VERSION}

WORKDIR /app

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get install -y python${PY_VERSION}
RUN ln -s /usr/bin/python${PY_VERSION} /usr/bin/python
RUN python3 -m pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install dist/capsa*

COPY test test
