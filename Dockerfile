FROM nvcr.io/nvidia/tensorrt:20.09-py3

# Install dependencies (1)
RUN apt-get update && apt-get install -y \
        automake autoconf libpng-dev nano \
        curl zip unzip libtool swig zlib1g-dev pkg-config git wget xz-utils \
        python3-mock libpython3-dev libpython3-all-dev \
        g++ gcc cmake make pciutils cpio gosu

# Install dependencies (2)
RUN pip3 install --upgrade pip \
    && pip3 install --upgrade tensorflowjs \
    && pip3 install --upgrade coremltools \
    && pip3 install --upgrade onnx \
    && pip3 install --upgrade tf2onnx \
    && pip3 install --upgrade tensorflow-datasets \
    && pip3 install --upgrade openvino2tensorflow \
    && pip3 install --upgrade onnxruntime \
    && pip3 install --upgrade onnx-simplifier \
    && pip3 install gdown \
    && ldconfig

# Install custom tflite_runtime, flatc, edgetpu-compiler
RUN gdown --id 1RWZmfFgtxm3muunv6BSf4yU29SKKFXIh \
    && chmod +x tflite_runtime-2.4.1-py3-none-any.whl \
    && pip3 install tflite_runtime-2.4.1-py3-none-any.whl \
    && rm tflite_runtime-2.4.1-py3-none-any.whl \
    && gdown --id 1drnpyrXkUHsMSqb8klV2YosEU9jdoJTP \
    && tar -zxvf flatc.tar.gz \
    && chmod +x flatc \
    && rm flatc.tar.gz \
    && wget https://github.com/PINTO0309/tflite2tensorflow/raw/main/schema/schema.fbs \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list \
    && apt-get update \
    && apt-get install edgetpu-compiler

# Install OpenVINO
ENV DEBIAN_FRONTEND=noninteractive

RUN gdown --id 1eaEV2loqNfnF06uhSPL-f4Sfue_5yh9o \
    && tar xf l_openvino_toolkit_p_2021.2.185.tgz \
    && rm l_openvino_toolkit_p_2021.2.185.tgz \
    && l_openvino_toolkit_p_2021.2.185/install_openvino_dependencies.sh \
    && sed -i 's/decline/accept/g' l_openvino_toolkit_p_2021.2.185/silent.cfg \
    && l_openvino_toolkit_p_2021.2.185/install.sh --silent l_openvino_toolkit_p_2021.2.185/silent.cfg \
    && source /opt/intel/openvino_2021/bin/setupvars.sh \
    && ${INTEL_OPENVINO_DIR}/install_dependencies/install_openvino_dependencies.sh \
    && sed -i 's/sudo -E //g' ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites.sh \
    && sed -i 's/tensorflow/#tensorflow/g' ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/requirements.txt \
    && sed -i 's/numpy/#numpy/g' ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/requirements.txt \
    && sed -i 's/onnx/#onnx/g' ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/requirements.txt \
    && ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites.sh \
    && rm -rf l_openvino_toolkit_p_2021.2.185 \
    && echo 'source /opt/intel/openvino_2021/bin/setupvars.sh' >> ${HOME}/.bashrc

# Install TensorRT additional package
RUN gdown --id 19vGQBrxJ-q6wMD995bgJq_GXqzaF8RhE \
    && dpkg -i nv-tensorrt-repo-ubuntu1804-cuda11.0-trt7.1.3.4-ga-20200617_1-1_amd64.deb \
    && apt-key add /var/nv-tensorrt-repo-cuda11.0-trt7.1.3.4-ga-20200617/7fa2af80.pub \
    && apt-get update \
    && apt-get install uff-converter-tf graphsurgeon-tf \
    && rm nv-tensorrt-repo-ubuntu1804-cuda11.0-trt7.1.3.4-ga-20200617_1-1_amd64.deb

# Install Custom TensorFlow (MediaPipe Custom OP, FlexDelegate, XNNPACK enabled)
RUN gdown --id 1nTSYsPXbZTIO2B7nIMtSpn5bBMlCr46N \
    && pip3 install --force-reinstall tensorflow-2.4.1-cp36-cp36m-linux_x86_64.whl \
    && rm tensorflow-2.4.1-cp36-cp36m-linux_x86_64.whl

WORKDIR /workspace

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]