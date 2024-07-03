ARG PYTHON_VERSION=3.11.6
ARG DEBIAN_VERSION=bookworm

FROM python:${PYTHON_VERSION}-slim-${DEBIAN_VERSION}
ARG PYTHON_VERSION
ARG DEBIAN_VERSION

ENV PYTHONUSERBASE=/opt/python

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY . /tmp/src
RUN pip install --no-cache-dir /tmp/src && rm -rf /tmp/src

ENTRYPOINT ["radifox-convert"]
