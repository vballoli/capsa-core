
# Dockerfile for the installed package

FROM ubuntu:latest

ARG PY_VERSION
ENV PY_VERSION ${PY_VERSION}
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get install -y python${PY_VERSION}
RUN ln -s /usr/bin/python${PY_VERSION} /usr/bin/python
RUN apt-get install -y python3-pip
RUN apt-get install -y python${PY_VERSION}-distutils
RUN python -m pip install --upgrade pip

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN python setup.py clean --all sdist
RUN pip install dist/capsa*

WORKDIR ./test
RUN python -m unittest test_ensemble.py -b
RUN python -m unittest test_mve.py -b
