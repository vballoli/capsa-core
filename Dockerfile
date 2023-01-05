
# Dockerfile for the installed package

FROM ubuntu:latest

ARG PY_VERSION
ENV PY_VERSION ${PY_VERSION}

WORKDIR /app

RUN apt update
RUN apt install python${PY_VERSION}
RUN python3 -m pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install dist/capsa*

COPY test test
