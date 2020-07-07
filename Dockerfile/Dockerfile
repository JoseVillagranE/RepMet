FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ARG PYTHON_VERSION=3.7

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64
ENV CUDNN_VERSION 7.6.5

RUN apt-get update && apt-get install -y apt-transport-https --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
	       wget \
	       nano \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
	       libcudnn7_$CUDNN_VERSION-1+cuda10.0 \
	       libcudnn7-dev_$CUDNN_VERSION-1+cuda10.0 && \
         rm -rf /var/lib/apt/lists/*


RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
     /opt/conda/bin/conda clean -ya

RUN conda create -y --name pytorch-py37 python=3.7.4 numpy pyyaml scipy ipython mkl pillow
RUN conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
RUN conda install -y mpi4py opencv
RUN pip install pandas hashfs pydevd remote_pdb rpdb matplotlib visdom
RUN pip install sacred GitPython pymongo tinydb tinydb-serialization tensorflow==1.3.0 pptree progressbar2 ipdb namedlist pyyaml cython

RUN pip install tensorboardX


RUN useradd -d /project -u 12567 --create-home user
USER user
WORKDIR /project/
ADD . /project/

ENV PYTHONPATH "$PYTHONPATH:/project/"

ENTRYPOINT ["/opt/conda/bin/python"]
