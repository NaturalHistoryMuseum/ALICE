# Base luigi Dockerfile with the latest Debian version and Python 3.5.2
FROM python:3.9-slim-bookworm

# Update the system and install required packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq-dev \
        build-essential \
        postgresql-client \
    && \
    useradd --password NP --create-home --shell /bin/bash luigi && \
    echo "luigi ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
        'luigi==3.3.0' \
        'sqlalchemy==1.1.0' \
        'psycopg2==2.9.1'

RUN mkdir /etc/luigi && chmod -R a+rw /etc/luigi

COPY scripts/ /
RUN chmod +x /generate_config.py && chmod +x /run.sh && mv /run.sh /usr/bin



USER luigi

CMD ["run.sh"]
