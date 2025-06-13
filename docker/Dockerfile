FROM python:3.9-slim-buster

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

WORKDIR $APP_HOME

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libmysqlclient-dev \
        git \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock* $APP_HOME/

RUN pip install poetry

RUN poetry install --no-dev --no-root

COPY . $APP_HOME

EXPOSE 8005

CMD ["poetry", "run", "uvicorn", "src.pipeline.server:app", "--host", "0.0.0.0", "--port", "8005"]