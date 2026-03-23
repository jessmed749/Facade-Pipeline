FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN git clone https://github.com/facebookresearch/segment-anything.git
RUN cd segment-anything && pip install -e .

COPY app /workspace/app
COPY scripts /workspace/scripts

RUN chmod +x /workspace/scripts/download_models.sh
RUN mkdir -p /workspace/checkpoints /workspace/input /workspace/output

CMD ["python3", "app/run_facade_pipeline.py"]