FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

RUN grep -v '[ -z "\$PS1" ] && return' ~/.bashrc >/tmp/bashrc && \
    mv /tmp/bashrc ~/.bashrc

# install core
COPY artifacts/docker/install/core.sh /install/core.sh
RUN bash /install/core.sh
# set cuda env
ENV PATH=/usr/local/cuda/bin:$PATH \
    CUDA_HOME=/usr/local/cuda/bin:$CUDA_HOME \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
# Install Conda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
ARG CONA_ENV=py311
RUN conda create -n $CONA_ENV python=3.11 -y
RUN echo "source activate $CONA_ENV" >> ~/.bashrc
ENV PATH=$CONDA_DIR/envs/$CONA_ENV/bin:$PATH

COPY artifacts/docker/install/python_package.sh /install/python_package.sh
RUN bash /install/python_package.sh
