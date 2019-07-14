# Copyright (c) 2019 Nobuo Tsukamoto
#
# This software is released under the MIT License.
# See the LICENSE file in the project root for more information.
#
# #==========================================================================

FROM tensorflow/tensorflow:1.13.1-gpu-py3

# Install wget (to make life easier below) and editors (to allow people to edit
# the files inside the container)
RUN apt-get update && \
    apt-get install -y wget git
RUN pip install pillow

# Get the tensorflow models research directory, and move it into tensorflow
# source folder to match recommendation of installation
RUN mkdir /tensorflow && cd /tensorflow && \
    git clone -b edge_tpu https://github.com/NobuoTsukamoto/models.git

# Set work direcotry and get dataset tools.
ENV TRAIN_DIR=/tensorflow/models/research/slim/transfer_learn/train/
ENV DATASET_DIR=/tensorflow/models/research/slim/transfer_learn/data/
ENV CKPT_DIR=/tensorflow/models/research/slim/transfer_learn/ckpt/
ARG work_dir=/tensorflow/models/research/slim
RUN cd ${work_dir} && \
    git clone https://github.com/NobuoTsukamoto/edge_tpu.git

WORKDIR ${work_dir}

