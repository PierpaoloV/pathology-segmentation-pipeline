# -----------------------------------------------------------------------
# CUDA 12.4 runtime (not devel) on Ubuntu 22.04 LTS
# Python 3.10 | PyTorch 2.4.1+cu124 | ASAP 2.2 Nightly
# -----------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Amsterdam

# -----------------------------------------------------------------------
# 1. System runtime libraries + Python 3.10 (Ubuntu 22.04 default)
#    Cleanup is inside the same RUN so the apt cache is never committed.
# -----------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # Python 3.10 (Ubuntu 22.04 default)
        python3-pip python3-dev python-is-python3 \
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
# 2. ASAP 2.2 Nightly (Ubuntu 22.04, Python 3.10 bindings)
#    curl -L follows redirects; apt-get install resolves deps correctly.
#    A .pth file registers /opt/ASAP/bin in Python's path reliably.
# -----------------------------------------------------------------------
ARG ASAP_URL=https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.2-(Nightly)/ASAP-2.2-Ubuntu2204.deb
RUN apt-get update && \
    curl -L "${ASAP_URL}" -o /tmp/ASAP.deb && \
    apt-get install --assume-yes /tmp/ASAP.deb && \
    SITE_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") && \
    printf "/opt/ASAP/bin/\n" > "${SITE_PACKAGES}/asap.pth" && \
    rm /tmp/ASAP.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------
# 3. Python packages — single layer, no pip cache
#    PyTorch is installed first (separate index-url) then the rest.
#    albumentations pinned to 1.4.x: 2.x has breaking API changes that
#    would require updates to training code before upgrading.
# -----------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools && \
    python3 -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.4.1 \
        torchvision==0.19.1 && \
    python3 -m pip install --no-cache-dir \
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


        httpx==0.27.2 \
        huggingface_hub==0.24.6 \
        opencv-python-headless>=4.6.0.66 \
        wholeslidedata==0.0.15 \
        openpyxl==3.1.5

# -----------------------------------------------------------------------
# 4. Non-root user
# -----------------------------------------------------------------------
RUN useradd -m -s /bin/bash user && \
    echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chown -R user:user /home/user/

# -----------------------------------------------------------------------
# 5. Source code
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
# 6. Environment
# -----------------------------------------------------------------------
ENV PYTHONPATH="/home/user/source/pathology-common:/home/user/source/pathology-fast-inference" \
    MPLBACKEND="Agg"

STOPSIGNAL SIGINT
EXPOSE 22 6006 8888

USER user
WORKDIR /home/user

ENTRYPOINT ["/home/user/execute.sh"]
