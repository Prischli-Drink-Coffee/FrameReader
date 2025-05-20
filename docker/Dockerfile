ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver
ARG BASE_IMAGE_TAG=24.01-py3

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS tritonserver-frame-reader

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        locales \
        lsof \
        psmisc \
        curl && \
    curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g pm2 && \
    locale-gen en_US.UTF-8 && \
    apt-get purge -y curl && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

RUN pip install uv

RUN uv venv /opt/.venv
ENV PATH="/opt/.venv/bin:$PATH"

COPY ./pyproject.toml /workspace/pyproject.toml

RUN pip install --pre --upgrade --extra-index-url https://pypi.nvidia.com tensorrt-cu12==10.4.0

RUN git clone https://github.com/NVIDIA/TensorRT.git -b release/10.4 --single-branch /tmp/TensorRT

RUN uv pip install -r /workspace/pyproject.toml

RUN uv pip install tritonclient[all]

RUN uv pip install "nvidia-modelopt[all]"

RUN mkdir -p /opt/tritonserver/backends/python
RUN mkdir -p /opt/tritonserver/backends/donut
RUN mkdir -p /opt/tritonserver/backends/yolo

RUN cp /opt/tritonserver/backends/python/libtriton_python.so /opt/tritonserver/backends/donut/
RUN cp /opt/tritonserver/backends/python/libtriton_python.so /opt/tritonserver/backends/yolo/

COPY ./models /workspace/models

ENTRYPOINT ["/bin/bash", "-c", "source /opt/.venv/bin/activate && exec \"$@\"", "--"]