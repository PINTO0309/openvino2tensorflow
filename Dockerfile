FROM ghcr.io/pinto0309/openvino2tensorflow:base.11.6.2-cudnn8-tf2.9.0rc0-trt8.4.0-openvino2021.4.582

ENV DEBIAN_FRONTEND=noninteractive
ARG APPVER
ARG WKDIR=/home/user
WORKDIR ${WKDIR}

# Install dependencies
RUN pip install --upgrade openvino2tensorflow \
    && pip install --upgrade tflite2tensorflow \
    && sudo ldconfig \
    && sudo pip cache purge
