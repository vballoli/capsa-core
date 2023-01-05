
# Dockerfile for the installed package

FROM ubuntu:latest

WORKDIR /app

RUN apt-get -y update
RUN apt-get install -y python${PY_VERSION}
RUN python -m pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install dist/capsa*

COPY test test
