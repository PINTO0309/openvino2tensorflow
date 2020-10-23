#!/bin/bash

python3 ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo_tf.py \
  --saved_model_dir saved_model \
  --output_dir openvino/reverse