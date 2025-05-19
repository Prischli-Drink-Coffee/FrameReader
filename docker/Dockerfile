ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver
ARG BASE_IMAGE_TAG=24.01-py3

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS tritonserver-frame-reader

RUN pip install uv

RUN uv venv /opt/.venv
ENV PATH="/opt/.venv/bin:$PATH"

COPY ./pyproject.toml /workspace/pyproject.toml

RUN pip install --pre --upgrade --extra-index-url https://pypi.nvidia.com tensorrt-cu12==10.4.0

RUN git clone https://github.com/NVIDIA/TensorRT.git -b release/10.4 --single-branch /tmp/TensorRT

RUN uv pip install -r /workspace/pyproject.toml

RUN uv pip install tritonclient[all]

RUN mkdir -p /opt/tritonserver/backends/python
RUN mkdir -p /opt/tritonserver/backends/donut
RUN mkdir -p /opt/tritonserver/backends/yolo

RUN cp /opt/tritonserver/backends/python/libtriton_python.so /opt/tritonserver/backends/donut/
RUN cp /opt/tritonserver/backends/python/libtriton_python.so /opt/tritonserver/backends/yolo/

COPY ./models /workspace/models

ENTRYPOINT ["/bin/bash", "-c", "source /opt/.venv/bin/activate && exec \"$@\"", "--"]