# We will use Ubuntu for our image
FROM nvidia/cuda:10.1-base-ubuntu18.04

# Updating Ubuntu packages
RUN apt-get update && \
    apt-get install -y wget ffmpeg build-essential bzip2

# Anaconda installing
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Set path to conda
ENV PATH /root/miniconda3/bin:$PATH

# Creates the conda environment
COPY env.yml .
RUN conda env create -f env.yml

# Initializes .bashrc with conda startup instructions
RUN /root/miniconda3/condabin/conda init bash && \
    /bin/bash -c "source ~/.bashrc"

RUN mkdir video-generation

# Configures bash to automatically start the environment
RUN echo "conda activate video-generation" >> ~/.bashrc

# Set the api key for wandb
ENV WANDB_API_KEY <YOUR WANDB API KEY>

WORKDIR video-generation
RUN /bin/bash -c "source ~/.bashrc && conda run -n video-generation wandb on"


# Run with docker run -it --gpus all --ipc=host -v /path/to/directory/video-generation:/video-generation video-generation:1.0 /bin/bash