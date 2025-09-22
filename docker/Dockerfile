FROM python:3.12-slim-bullseye

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

WORKDIR $APP_HOME

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        default-mysql-client \
        build-essential \
        git \
        ffmpeg \
        netcat \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml $APP_HOME/

RUN pip install uv

RUN uv sync

COPY db.sh /usr/local/bin/db.sh
RUN chmod +x /usr/local/bin/db.sh

COPY server.sh $APP_HOME/server.sh
RUN chmod +x $APP_HOME/server.sh

COPY . $APP_HOME
