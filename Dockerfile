# ============================================================================
# Unified Pathology Pipeline — CUDA 12.8.1 | Python 3.11
# Includes: pathology-segmentation-pipeline + slide2vec 4.3.0 + CellViT++
#
# Strategy:
# - Reuse slide2vec's upstream CUDA 12.8.1 / Python 3.11 container baseline.
# - Install slide2vec from the vendored ./slide2vec tree with the full fm extra.
# - Keep segmentation-models-pytorch on the timm-compatible 0.5.x line.
# - Add the CellViT++ dependency layer on top of the same Torch/CUDA stack.
#
# Notes:
# - CUDA-bound packages are installed in the build stage and copied to runtime.
# - We keep numpy on a 1.x line to satisfy slide2vec and older ML dependencies.
# - TensorFlow is retained for CellViT++ utility paths, but left on a newer
#   Python 3.11 compatible line rather than the older 2.12 pin.
# ============================================================================

ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.8.1
ARG PYTORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG ASAP_URL=https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.2-(Nightly)/ASAP-2.2-Ubuntu2204.deb
ARG LIBJPEG_TURBO_VERSION=3.1.0

########################
# Stage 1: build stage #
########################
FROM --platform=linux/amd64 nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS build

ARG PYTORCH_CUDA_INDEX_URL
ARG ASAP_URL
ARG LIBJPEG_TURBO_VERSION

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Amsterdam

WORKDIR /opt/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        git \
        gnupg2 \
        gpg-agent \
        libnuma1 \
        libopenjp2-7-dev \
        libsnappy-dev \
        libspatialindex-dev \
        libtiff-dev \
        libvips-dev \
        ninja-build \
        openssh-server \
        software-properties-common \
        vim \
        screen \
        zip \
        unzip \
        zlib1g-dev \
    && mkdir -p /var/run/sshd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3.11-venv \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${LIBJPEG_TURBO_VERSION}/libjpeg-turbo-${LIBJPEG_TURBO_VERSION}.tar.gz \
      | tar xz -C /tmp \
    && cd /tmp/libjpeg-turbo-${LIBJPEG_TURBO_VERSION} \
    && cmake -G"Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/usr/local . \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig \
    && rm -rf /tmp/libjpeg-turbo-${LIBJPEG_TURBO_VERSION}

RUN python -m ensurepip --upgrade \
    && python -m pip install --upgrade pip setuptools pip-tools \
    && python -m pip install hatchling psutil \
    && rm -rf /root/.cache/pip

RUN printf '%s\n' \
    'numpy<2' \
    > /opt/app/constraints-cu128.txt

RUN python -m pip install --no-cache-dir --no-color \
    -c /opt/app/constraints-cu128.txt \
    --extra-index-url "${PYTORCH_CUDA_INDEX_URL}" \
    "slide2vec[fm]==4.3.0"

RUN python -m pip install --no-cache-dir --no-color \
    -c /opt/app/constraints-cu128.txt \
    --extra-index-url "${PYTORCH_CUDA_INDEX_URL}" \
    git+https://github.com/lilab-stanford/MUSK.git \
    git+https://github.com/Mahmoodlab/CONCH.git \
    git+https://github.com/prov-gigapath/prov-gigapath.git \
    git+https://github.com/facebookresearch/sam2.git

RUN python -m pip install --no-cache-dir --no-color \
    -c /opt/app/constraints-cu128.txt \
    --extra-index-url "${PYTORCH_CUDA_INDEX_URL}" \
    'flash-attn>=2.7.1,<=2.8.0' \
    --no-build-isolation

RUN python -m pip install --no-cache-dir --no-color \
    -c /opt/app/constraints-cu128.txt \
    --extra-index-url "${PYTORCH_CUDA_INDEX_URL}" \
        albumentations==1.3.1 \
    cupy-cuda12x \
    cucim-cu12 \
    geojson==3.0.1 \
    h5py==3.11.0 \
    httpx==0.27.2 \
    ipykernel \
    jupyterlab==4.2.5 \
    natsort==8.4.0 \
    numpy==1.23.5 \
    opencv-python-headless==4.10.0.84 \
    openpyxl==3.1.5 \
    openslide-python==1.3.1 \
    pandarallel==1.6.5 \
    pathopatch==1.0.2 \
    pyaml==24.7.0 \
    pyjwt==2.6.0 \
    pyvips==2.2.3 \
    rasterio==1.3.5.post1 \
    ray==2.9.3 \
    rdp==0.8 \
    schema==0.7.5 \
    seaborn==0.13.2 \
    segmentation-models-pytorch==0.5.0 \
    simpleitk==2.3.1 \
    tensorflow-cpu==2.17.1 \
    torchaudio \
    torchinfo==1.8.0 \
    torchmetrics==0.11.4 \
    torchstain==1.3.0 \
    tqdm==4.66.5 \
    ujson==5.10.0 \
    wandb==0.17.9 \
    wholeslidedata==0.0.15 \
    wsidicom==0.20.4 \
    wsidicomizer==0.14.1 \
    xgboost==2.1.1 \
    scikit-base==0.7.8 \
    cachetools==5.3.3 \
    colorama==0.4.6 \
    future==0.18.2 \
    flatbuffers==24.3.25 \
    opt-einsum==3.3.0 \
    pydantic==1.10.4 \
    pydicom==2.4.4 \
    python-snappy==0.7.3 \
    tabulate==0.9.0 \
    termcolor==2.4.0 \
    evalutils==0.4.2 \
    colour==0.1.5

##########################
# Stage 2: runtime stage #
##########################
FROM --platform=linux/amd64 nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION}

ARG ASAP_URL

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Amsterdam

WORKDIR /home/user
ENV PATH="/home/user/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        libboost-filesystem1.74.0 \
        libboost-iostreams1.74.0 \
        libboost-regex1.74.0 \
        libboost-thread1.74.0 \
        libexif12 \
        libfftw3-3 \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libgsf-1-114 \
        libnuma1 \
        libopenjp2-7-dev \
        libopenslide0 \
        librsvg2-2 \
        libsnappy-dev \
        libtiff5 \
        libtiff-dev \
        libqt5concurrent5 \
        libqt5core5a \
        libqt5gui5 \
        libqt5widgets5 \
        libvips-dev \
        openssh-server \
        pv \
        screen \
        software-properties-common \
        sudo \
        vim \
        zip \
        unzip \
        zlib1g-dev \
    && mkdir -p /var/run/sshd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3.11-venv \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/lib/libjpeg* /usr/local/lib/
COPY --from=build /usr/local/lib/libturbojpeg* /usr/local/lib/
RUN ldconfig

RUN apt-get update \
    && curl -L "${ASAP_URL}" -o /tmp/ASAP.deb \
    && apt-get install --assume-yes /tmp/ASAP.deb \
    && SITE_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
    && printf "/opt/ASAP/bin/\n" > "${SITE_PACKAGES}/asap.pth" \
    && rm -f /tmp/ASAP.deb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY --from=build /usr/local/share/jupyter /usr/local/share/jupyter
COPY --from=build /usr/local/etc/jupyter /usr/local/etc/jupyter

RUN echo "/usr/local/lib/python3.11/dist-packages/nvidia/nvimgcodec" > /etc/ld.so.conf.d/nvimgcodec.conf \
    && ldconfig

RUN useradd -m -s /bin/bash user \
    && echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && chown -R user:user /home/user/

COPY --chown=user:user pathology-common /home/user/source/pathology-common
COPY --chown=user:user pathology-fast-inference /home/user/source/pathology-fast-inference
COPY --chown=user:user code /home/user/source/code
COPY --chown=user:user download_models.py /home/user/source/download_models.py
COPY --chown=user:user execute.sh /home/user/execute.sh

RUN mkdir -p /home/user/source/models \
    && chown -R user:user /home/user/source/models

ENV PYTHONPATH="/home/user/source/pathology-common:/home/user/source/pathology-fast-inference" \
    MPLBACKEND="Agg"

STOPSIGNAL SIGINT
EXPOSE 22 6006 8888

USER user
WORKDIR /home/user

ENTRYPOINT ["/home/user/execute.sh"]
