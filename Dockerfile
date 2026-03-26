# -----------------------------------------------------------------------
# CUDA 12.4 runtime (not devel) on Ubuntu 22.04 LTS
# Python 3.11 | PyTorch 2.4.1+cu124 | ASAP 2.2 Nightly
# -----------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Amsterdam

# -----------------------------------------------------------------------
# 1. System runtime libraries + Python 3.11
#    Cleanup is inside the same RUN so the apt cache is never committed.
# -----------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        # Python 3.11
        python3.11 python3.11-dev \
        # Utilities
        curl git openssh-server sudo pv \
        # OpenGL / threading (PyTorch, OpenCV, ASAP)
        libgl1 libgomp1 \
        # Image I/O (OpenSlide / ASAP)
        libopenslide0 libtiff5 libjpeg-turbo8 \
        # ASAP Qt5 runtime
        libqt5concurrent5 libqt5core5a libqt5gui5 libqt5widgets5 \
        # ASAP Boost runtime (specific components — not libboost-all-dev)
        libboost-filesystem1.74.0 libboost-regex1.74.0 \
        libboost-thread1.74.0 libboost-iostreams1.74.0 \
        # ASAP misc runtime
        libglib2.0-0 libgsf-1-114 libexif12 librsvg2-2 libfftw3-3 && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------
# 2. Bootstrap pip for Python 3.11 and set as default python3
# -----------------------------------------------------------------------
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# -----------------------------------------------------------------------
# 3. ASAP 2.2 Nightly (Ubuntu 22.04)
#    dpkg -i ... || apt-get -f install -y  resolves missing deps correctly
#    instead of silently swallowing failures with || true.
# -----------------------------------------------------------------------
ARG ASAP_URL=https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.2-(Nightly)/ASAP-2.2-Ubuntu2204.deb
RUN curl -fsSL -o /tmp/ASAP.deb "${ASAP_URL}" && \
    dpkg -i /tmp/ASAP.deb || apt-get install -f -y && \
    rm /tmp/ASAP.deb && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------
# 4. Python packages — single layer, no pip cache
#    PyTorch is installed first (separate index-url) then the rest.
#    albumentations pinned to 1.4.x: 2.x has breaking API changes that
#    would require updates to training code before upgrading.
# -----------------------------------------------------------------------
RUN python3.11 -m pip install --no-cache-dir \
        torch==2.4.1 \
        torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    python3.11 -m pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.2.2 \
        matplotlib==3.9.0 \
        scikit-learn==1.5.1 \
        scikit-image==0.24.0 \
        scipy==1.13.1 \
        segmentation-models-pytorch==0.3.4 \
        albumentations==1.4.14 \
        wandb==0.17.9 \
        rich==13.7.1 \
        shapely==2.0.6 \
        rdp==0.8 \
        jupyterlab==4.2.5 \
        huggingface_hub==0.24.6 \
        # Pre-install headless OpenCV to win the conflict resolution when slide2vec
        # is installed later: hs2p requires opencv-python (full) while wholeslidedata
        # requires opencv-python-headless. Both conflict; headless is correct for
        # server/Docker use. Pre-installing it here makes pip treat it as satisfying
        # the opencv-python requirement from hs2p.
        opencv-python-headless>=4.6.0.66 \
        # wholeslidedata pinned to <0.0.16 to match the slide2vec constraint.
        wholeslidedata==0.0.15

# -----------------------------------------------------------------------
# 5. Non-root user
# -----------------------------------------------------------------------
RUN useradd -m -s /bin/bash user && \
    echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chown -R user:user /home/user/

# -----------------------------------------------------------------------
# 6. Source code
#    Models are NOT baked in — they are downloaded from HuggingFace Hub
#    at container startup by download_models.py (see execute.sh).
#    Mount a host directory to /home/user/source/models to cache them
#    across container restarts:
#      docker run -v /host/path/models:/home/user/source/models ...
# -----------------------------------------------------------------------
COPY --chown=user:user pathology-common         /home/user/source/pathology-common
COPY --chown=user:user pathology-fast-inference /home/user/source/pathology-fast-inference
COPY --chown=user:user code                     /home/user/source/code
COPY --chown=user:user download_models.py       /home/user/source/download_models.py
COPY --chown=user:user execute.sh               /home/user/execute.sh

# Create the models directory so the download script can write into it
RUN mkdir -p /home/user/source/models && \
    chown -R user:user /home/user/source/models

# -----------------------------------------------------------------------
# 7. Environment
# -----------------------------------------------------------------------
ENV PYTHONPATH="/opt/ASAP/bin:/home/user/source/pathology-common:/home/user/source/pathology-fast-inference" \
    MPLBACKEND="Agg"

STOPSIGNAL SIGINT
EXPOSE 22 6006 8888

USER user
WORKDIR /home/user

ENTRYPOINT ["/home/user/execute.sh"]
