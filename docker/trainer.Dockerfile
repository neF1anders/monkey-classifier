FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ARG UID
ARG GID

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN groupadd -g ${GID} group && \
    useradd -m -u ${UID} -g group monkey

WORKDIR /workspace
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

USER monkey

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
