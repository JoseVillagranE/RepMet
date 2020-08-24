FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
ARG PYTHON_VERSION=3.7

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64
ENV CUDNN_VERSION 7.6.5.32

RUN apt-get update && apt-get install -y apt-transport-https --no-install-recommends --allow-downgrades --allow-change-held-packages \
         build-essential \
         cmake \
         git \
         curl \
	       wget \
	       nano \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
	       libcudnn7=$CUDNN_VERSION-1+cuda10.1 \
	       libcudnn7-dev=$CUDNN_VERSION-1+cuda10.1 && \
         rm -rf /var/lib/apt/lists/*


RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH
RUN conda create -y --name pytorch-py37 python=3.7.4 numpy pyyaml scipy ipython mkl pillow
RUN conda install pytorch==1.5.0 torchvision cudatoolkit=10.1 -c pytorch
RUN conda install -y mpi4py opencv
RUN pip install torchvision==0.6.0
RUN pip install easydict
RUN pip install pandas hashfs pydevd remote_pdb rpdb matplotlib visdom
RUN pip install sacred GitPython pymongo tinydb tinydb-serialization tensorflow==1.15.3 pptree progressbar2 ipdb namedlist pyyaml cython

RUN pip install tensorboardX
RUN pip install sklearn

RUN useradd -d /Repmet -u 12567 --create-home user
COPY . /Repmet/

#USER user
WORKDIR /Repmet
#ADD . /project/

ENV PYTHONPATH "$PYTHONPATH:/Repmet/"

ENTRYPOINT ["/opt/conda/bin/python"]
