# Base luigi Dockerfile with the latest Debian version and Python 3.5.2
FROM python:3.10-slim-bookworm

# Update the system and install required packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq-dev \
        build-essential \
        postgresql-client \
        git \
        nano \
        ffmpeg \
        libsm6 \
        libxext6 \
    && \
    useradd --password NP --create-home --shell /bin/bash luigi && \
    echo "luigi ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel setuptools requests

# Torch & wheel need to be available before detectron2 (in requirements)
# postgres libs are only needed for luigi scheduler
RUN python -m pip install --no-cache-dir \
        'luigi==3.3.0' \
        'sqlalchemy==1.4.49' \
        'psycopg2==2.9.1' \ 
        'wheel==0.41.2' \
        'torch==2.0.1'

RUN mkdir /etc/luigi && chmod -R a+rw /etc/luigi

COPY ./docker/scripts/ /
RUN chmod +x /generate_config.py && chmod +x /run.sh && mv /run.sh /usr/bin

RUN mkdir /alice

COPY ./requirements.txt /alice/
COPY ./setup.py /alice/
COPY ./alice /alice/alice
COPY ./scripts/ /alice/scripts
COPY ./data/models/ /alice/data/models

RUN chown -R luigi:luigi /alice
RUN chmod 755 /alice

WORKDIR /alice
RUN pip install -r ./requirements.txt
RUN python ./setup.py develop
WORKDIR /

USER luigi
CMD ["run.sh"]
