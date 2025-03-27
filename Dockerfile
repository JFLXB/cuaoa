FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=24.9.2-0
ARG TARGETPLATFORM

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN apt update && apt upgrade -y && \
    apt install -y --no-install-recommends \
    curl \
    git \
    wget && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes  && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

RUN . ${CONDA_DIR}/etc/profile.d/conda.sh && conda config --set auto_activate_base false
RUN . ${CONDA_DIR}/etc/profile.d/conda.sh && conda create -n cuaoa python=3.11 -y

RUN echo "conda deactivate" >> ~/.bashrc
RUN echo "source ~/cuaoa/.venv/bin/activate" >> ~/.bashrc

RUN echo "\necho \"============================\nCUAOA License and Compliance\n============================\n\nThe CUAOA project is licensed under the Apache License 2.0. See the /LICENSE file for details.\nAlternatively the LICENSE can be obtained here: https://github.com/JFLXB/cuaoa/blob/main/LICENSE\n\nBy using this software, you agree to comply with the licenses of all dependencies used in this project.\nNotably, the cuStateVec library has its own licensing terms which must be adhered to:\nhttps://docs.nvidia.com/cuda/cuquantum/latest/license.html\n\"" >> /etc/bash.bashrc

COPY . /root/cuaoa
RUN cp /root/cuaoa/LICENSE /root/LICENSE

WORKDIR /root/cuaoa
RUN . ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate cuaoa && ./install.sh --verbose
RUN . ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate cuaoa && uv sync

WORKDIR /root

CMD ["/bin/bash"]
