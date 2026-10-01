FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Intel GPU user-space driver (Level Zero + OpenCL) with Battlemage support.
RUN apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common gpg-agent ca-certificates \
 && add-apt-repository -y ppa:kobuk-team/intel-graphics \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      libze1 libze-intel-gpu1 intel-opencl-icd intel-gsc clinfo \
      python3.12 python3.12-venv git \
 && rm -rf /var/lib/apt/lists/*

# Match the host user's UID/GID so the mounted HF cache and outputs stay owned by you.
ARG UID=1000
ARG GID=1000
RUN userdel -r ubuntu 2>/dev/null || true \
 && groupadd -g ${GID} app \
 && useradd -m -u ${UID} -g ${GID} app

ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN python3.12 -m venv /opt/venv \
 && pip install --upgrade pip \
 && pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu \
 && pip install "transformers>=5.17" accelerate pillow gradio sentencepiece \
      "git+https://github.com/huggingface/diffusers" \
 && pip freeze > /opt/requirements.lock

WORKDIR /app
COPY app.py /app/app.py
RUN mkdir -p /app/outputs /home/app/.cache/huggingface && chown -R app:app /app /home/app

USER app
EXPOSE 7860
CMD ["python", "app.py"]
