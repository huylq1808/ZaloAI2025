FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /src

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common wget curl build-essential ca-certificates \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3.9 python3.9-dev python3.9-venv \
       python3-pip python3-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && python -m pip install --upgrade pip setuptools wheel

#TẠO CONSTRAINTS
RUN printf "numpy==1.26.4\n" > /tmp/constraints.txt

# (gỡ numpy nếu có trước đó
RUN pip uninstall -y numpy || true

# ---  TORCH (với constraint) ---
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
    torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

#ULTRALYTICS + các thư viện
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
    lap>=0.5.12 \
    ultralytics opencv-python-headless Pillow tqdm jupyter ipykernel

# verify 
RUN python - <<'PY'
import sys, numpy, torch
print("PYTHON:", sys.executable, sys.version.splitlines()[0])
print("NUMPY:", numpy.__version__, "->", numpy.__file__)
print("TORCH:", torch.__version__, "->", torch.__file__)
try:
    import ultralytics
    print("ULTRALYTICS:", ultralytics.__version__, "->", ultralytics.__file__)
    import lap
    print("LAP:", lap.__version__)
except Exception as e:
    print("IMPORT ERROR:", e)
PY

RUN python -m ipykernel install --user --name python39 --display-name "Python 3.9 (YOLO)"

COPY ./code /src/code 

EXPOSE 8888
CMD ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0", "--no-browser", "--port=8888"]
