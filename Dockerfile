FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ARG OSVER=ubuntu2004
ARG TENSORFLOWVER=2.8.0
ARG CPVER=cp38
ARG OPENVINOVER=2021.4.582
ARG OPENVINOROOTDIR=/opt/intel/openvino_2021
ARG TENSORRTVER=cuda11.4-trt8.2.3.0-ga-20220113
ARG ONNXRUNTIMEVER=1.11.0
ARG APPVER
ARG WKDIR=/home/user

# dash -> bash
RUN echo "dash dash/sh boolean false" | debconf-set-selections \
    && dpkg-reconfigure -p low dash
COPY bashrc ${WKDIR}/.bashrc
WORKDIR ${WKDIR}

# Install dependencies (1)
RUN apt-get update && apt-get install -y \
        automake autoconf libpng-dev nano python3-pip \
        curl zip unzip libtool swig zlib1g-dev pkg-config \
        python3-mock libpython3-dev libpython3-all-dev \
        g++ gcc make pciutils cpio gosu wget libmkldnn-dev \
        libgtk-3-dev libxtst-dev sudo apt-transport-https \
        build-essential gnupg git xz-utils vim libyaml-cpp-dev \
        libva-drm2 libva-x11-2 vainfo libva-wayland2 libva-glx2 \
        libva-dev libdrm-dev xorg xorg-dev protobuf-compiler \
        openbox libx11-dev libgl1-mesa-glx libgl1-mesa-dev \
        libtbb2 libtbb-dev libopenblas-dev libopenmpi-dev \
        python-is-python3 software-properties-common \
    && sed -i 's/# set linenumbers/set linenumbers/g' /etc/nanorc \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies (2)
RUN pip3 install --upgrade pip \
    && pip install --upgrade numpy==1.19.5 \
    && pip install --upgrade tensorflowjs \
    && pip install --upgrade coremltools \
    && pip install --upgrade paddlepaddle \
    && pip install --upgrade lap \
    && pip install --upgrade pycocotools \
    && pip install --upgrade scipy \
    && pip install --upgrade paddle2onnx \
    && pip install --upgrade onnx \
    && pip install --upgrade onnxruntime-extensions \
    && pip install --upgrade onnx-simplifier \
    && pip install --upgrade onnxmltools \
    && pip install --upgrade onnxconverter-common \
    && pip install --upgrade tf2onnx \
    && pip install --upgrade onnx-tf \
    && pip install --upgrade tensorflow-datasets \
    && pip install --upgrade openvino2tensorflow \
    && pip install --upgrade tflite2tensorflow \
    && pip install --upgrade gdown \
    && pip install --upgrade PyYAML \
    && pip install --upgrade matplotlib \
    && pip install --upgrade tf_slim \
    && pip install --upgrade pandas \
    && pip install --upgrade numexpr \
    && pip install --upgrade onnx2json \
    && pip install --upgrade json2onnx \
    && pip install --upgrade gluoncv \
    && pip install --upgrade dgl \
    && pip install --upgrade cmake \
    && pip install --upgrade ninja \
    && python3 -m pip install onnx_graphsurgeon \
        --index-url https://pypi.ngc.nvidia.com \
    && pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 \
        -f https://download.pytorch.org/whl/cu113/torch_stable.html \
    && pip install pycuda==2021.1 \
    && pip install scikit-image \
    && pip install performance-monitor \
    && ldconfig \
    && pip cache purge \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Install sclblonnx non-version check custom .ver
RUN wget https://github.com/PINTO0309/openvino2tensorflow/releases/download/${APPVER}/sclblonnx-0.1.9_nvc-py3-none-any.whl \
    && pip3 install sclblonnx-0.1.9_nvc-py3-none-any.whl \
    && rm sclblonnx-0.1.9_nvc-py3-none-any.whl \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Install custom tflite_runtime, flatc, edgetpu-compiler
RUN wget https://github.com/PINTO0309/openvino2tensorflow/releases/download/${APPVER}/tflite_runtime-${TENSORFLOWVER}-${CPVER}-none-linux_x86_64.whl \
    && chmod +x tflite_runtime-${TENSORFLOWVER}-${CPVER}-none-linux_x86_64.whl \
    && pip3 install --force-reinstall tflite_runtime-${TENSORFLOWVER}-${CPVER}-none-linux_x86_64.whl \
    && rm tflite_runtime-${TENSORFLOWVER}-${CPVER}-none-linux_x86_64.whl \
    && wget https://github.com/PINTO0309/openvino2tensorflow/releases/download/${APPVER}/flatc.tar.gz \
    && tar -zxvf flatc.tar.gz \
    && chmod +x flatc \
    && rm flatc.tar.gz \
    && wget https://github.com/PINTO0309/tflite2tensorflow/raw/main/schema/schema.fbs \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list \
    && apt-get update \
    && apt-get install edgetpu-compiler \
    && pip cache purge \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Install OpenVINO
RUN wget https://github.com/PINTO0309/openvino2tensorflow/releases/download/${APPVER}/l_openvino_toolkit_p_${OPENVINOVER}.tgz \
    && tar xf l_openvino_toolkit_p_${OPENVINOVER}.tgz \
    && rm l_openvino_toolkit_p_${OPENVINOVER}.tgz \
    && l_openvino_toolkit_p_${OPENVINOVER}/install_openvino_dependencies.sh -y \
    && sed -i 's/decline/accept/g' l_openvino_toolkit_p_${OPENVINOVER}/silent.cfg \
    && l_openvino_toolkit_p_${OPENVINOVER}/install.sh --silent l_openvino_toolkit_p_${OPENVINOVER}/silent.cfg \
    && source ${OPENVINOROOTDIR}/bin/setupvars.sh \
    && ${INTEL_OPENVINO_DIR}/install_dependencies/install_openvino_dependencies.sh \
    && sed -i 's/sudo -E //g' ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites.sh \
    && sed -i 's/tensorflow/#tensorflow/g' ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/requirements.txt \
    && sed -i 's/numpy/#numpy/g' ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/requirements.txt \
    && sed -i 's/onnx/#onnx/g' ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/requirements.txt \
    && ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites.sh \
    && rm -rf l_openvino_toolkit_p_${OPENVINOVER} \
    && echo "source ${OPENVINOROOTDIR}/bin/setupvars.sh" >> .bashrc \
    && echo "${OPENVINOROOTDIR}/deployment_tools/ngraph/lib/" >> /etc/ld.so.conf \
    && echo "${OPENVINOROOTDIR}/deployment_tools/inference_engine/lib/intel64/" >> /etc/ld.so.conf \
    && pip cache purge \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Install TensorRT additional package
RUN wget https://github.com/PINTO0309/openvino2tensorflow/releases/download/${APPVER}/nv-tensorrt-repo-${OSVER}-${TENSORRTVER}_1-1_amd64.deb \
    && dpkg -i nv-tensorrt-repo-${OSVER}-${TENSORRTVER}_1-1_amd64.deb \
    && apt-key add /var/nv-tensorrt-repo-${OSVER}-${TENSORRTVER}/7fa2af80.pub \
    && apt-get update \
    && apt-get install -y \
        tensorrt uff-converter-tf graphsurgeon-tf \
        python3-libnvinfer-dev onnx-graphsurgeon \
    && rm nv-tensorrt-repo-${OSVER}-${TENSORRTVER}_1-1_amd64.deb \
    && cd /usr/src/tensorrt/samples/trtexec \
    && make \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# onnxruntime-gpu (CUDA, TensorRT)
# https://zenn.dev/pinto0309/scraps/1e9c0a00112cf9
RUN pip uninstall -y onnxruntime onnxruntime-gpu \
    && wget https://github.com/PINTO0309/openvino2tensorflow/releases/download/${APPVER}/onnxruntime_gpu-${ONNXRUNTIMEVER}-${CPVER}-none-linux_x86_64.whl \
    && pip install onnxruntime_gpu-${ONNXRUNTIMEVER}-${CPVER}-none-linux_x86_64.whl \
    && rm onnxruntime_gpu-${ONNXRUNTIMEVER}-${CPVER}-none-linux_x86_64.whl

# Install Custom TensorFlow (MediaPipe Custom OP, FlexDelegate, XNNPACK enabled)
# https://github.com/PINTO0309/Tensorflow-bin
RUN wget https://github.com/PINTO0309/openvino2tensorflow/releases/download/${APPVER}/tensorflow-${TENSORFLOWVER}-${CPVER}-none-linux_x86_64.whl \
    && pip3 install --force-reinstall tensorflow-${TENSORFLOWVER}-${CPVER}-none-linux_x86_64.whl \
    && rm tensorflow-${TENSORFLOWVER}-${CPVER}-none-linux_x86_64.whl \
    && pip cache purge \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Install onnx-tensorrt
RUN git clone --recursive https://github.com/onnx/onnx-tensorrt \
    && cd onnx-tensorrt \
    && git checkout 1f041ce6d7b30e9bce0aacb2243309edffc8fb3c \
    && mkdir build \
    && cd build \
    && cmake .. -DTENSORRT_ROOT=/usr/src/tensorrt \
    && make -j$(nproc) \
    && make install

# Install torch2trt
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt \
    && cd torch2trt \
    && git checkout 0400b38123d01cc845364870bdf0a0044ea2b3b2 \
    # https://github.com/NVIDIA-AI-IOT/torch2trt/issues/619
    && wget https://github.com/NVIDIA-AI-IOT/torch2trt/commit/8b9fb46ddbe99c2ddf3f1ed148c97435cbeb8fd3.patch \
    && git apply 8b9fb46ddbe99c2ddf3f1ed148c97435cbeb8fd3.patch \
    && python3 setup.py install

# Download the ultra-small sample data set for INT8 calibration
RUN mkdir sample_npy \
    && wget -O sample_npy/calibration_data_img_sample.npy https://github.com/PINTO0309/openvino2tensorflow/releases/download/${APPVER}/calibration_data_img_sample.npy

# LLVM
RUN wget https://apt.llvm.org/llvm.sh \
    && chmod +x llvm.sh \
    && ./llvm.sh 14 \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# # NNPACK
# RUN git clone --recursive https://github.com/Maratyszcza/NNPACK.git \
#     && cd NNPACK \
#     && git checkout c07e3a0400713d546e0dea2d5466dd22ea389c73 \
#     && sed -i "s|gnu99|gnu99 -fPIC|g" CMakeLists.txt \
#     && sed -i "s|gnu++11|gnu++11 -fPIC|g" CMakeLists.txt \
#     && mkdir build \
#     && cd build \
#     && cmake -G Ninja -D BUILD_SHARED_LIBS=ON .. \
#     && ninja \
#     && ninja install \
#     && sh -c "echo '/usr/local/lib' > /etc/ld.so.conf.d/nnpack.conf" \
#     && ldconfig

# Clear caches
RUN apt clean \
    && rm -rf /var/lib/apt/lists/*

# Create a user who can sudo in the Docker container
ENV USERNAME=user
RUN echo "root:root" | chpasswd \
    && adduser --disabled-password --gecos "" "${USERNAME}" \
    && echo "${USERNAME}:${USERNAME}" | chpasswd \
    && echo "%${USERNAME}    ALL=(ALL)   NOPASSWD:    ALL" >> /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}
USER ${USERNAME}
RUN sudo chown ${USERNAME}:${USERNAME} ${WKDIR} \
    && sudo chmod 777 ${WKDIR}/.bashrc

# OpenCL settings - https://github.com/intel/compute-runtime/releases
RUN cd ${OPENVINOROOTDIR}/install_dependencies/ \
    && yes | sudo -E ./install_NEO_OCL_driver.sh \
    && cd ${WKDIR} \
    && wget https://github.com/intel/compute-runtime/releases/download/21.35.20826/intel-gmmlib_21.2.1_amd64.deb \
    && wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.8517/intel-igc-core_1.0.8517_amd64.deb \
    && wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.8517/intel-igc-opencl_1.0.8517_amd64.deb \
    && wget https://github.com/intel/compute-runtime/releases/download/21.35.20826/intel-opencl_21.35.20826_amd64.deb \
    && wget https://github.com/intel/compute-runtime/releases/download/21.35.20826/intel-ocloc_21.35.20826_amd64.deb \
    && wget https://github.com/intel/compute-runtime/releases/download/21.35.20826/intel-level-zero-gpu_1.2.20826_amd64.deb \
    && sudo dpkg -i *.deb \
    && rm *.deb \
    && sudo apt clean \
    && sudo rm -rf /var/lib/apt/lists/*

# TVM
RUN git clone --recursive https://github.com/apache/tvm \
    && cd tvm \
    # 0.9.dev0
    && git checkout faa2e6ad578f09b1248b5303b17c3fa24ebd0bd6 \
    && git submodule init \
    && git submodule update \
    && mkdir build \
    && cd build \
    && cmake \
        -DUSE_CUDA=ON \
        -DUSE_MICRO=OFF \
        -DUSE_MICRO_STANDALONE_RUNTIME=OFF \
        -DUSE_CPP_RPC=ON \
        -DUSE_GRAPH_EXECUTOR_CUDA_GRAPH=ON \
        -DUSE_PIPELINE_EXECUTOR=ON \
        -DUSE_RPC=ON \
        -DUSE_LLVM=ON \
        -DUSE_MKLDNN=ON \
        -DUSE_OPENMP=ON \
        -DUSE_NNPACK=OFF \
        # -DUSE_TFLITE=/usr/local/lib/libtensorflow-lite.a \
        # -DUSE_EDGETPU=OFF \
        -DUSE_CUDNN=ON \
        -DUSE_TENSORRT_CODEGEN=ON \
        -DUSE_TENSORRT_RUNTIME=ON \
        -DUSE_TF_TVMDSOOP=ON \
        -DUSE_PT_TVMDSOOP=ON \
        -DUSE_TARGET_ONNX=ON \
        -DBUILD_STATIC_RUNTIME=ON \
        -DSUMMARIZE=ON \
        .. \
        -G Ninja \
    && ninja \
    && sudo ninja install

# TVM Python binding
RUN cd ${HOME}/tvm/python \
    && python setup.py install --user

# Final processing of onnx-tensorrt install
RUN echo 'GPU=$(python3 -c "import torch;print(torch.cuda.is_available())")' >> ${HOME}/.bashrc \
    && echo 'if [ $GPU = "True" ]; then' >> ${HOME}/.bashrc \
    && echo "export PATH=${PATH}:/usr/src/tensorrt/bin:/onnx-tensorrt/build" >> ${HOME}/.bashrc \
    && echo "cd ${HOME}/onnx-tensorrt" >> ${HOME}/.bashrc \
    && echo "sudo python setup.py install" >> ${HOME}/.bashrc \
    && echo "fi" >> ${HOME}/.bashrc \
    && echo "cd ${WKDIR}" >> ${HOME}/.bashrc \
    && echo "cd ${HOME}/workdir" >> ${HOME}/.bashrc
