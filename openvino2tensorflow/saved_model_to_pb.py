#! /usr/bin/env python
### tensorflow==2.3.1

import os
import logging
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config
import shutil
import argparse


def convert_saved_model_to_graph_def(saved_model_dir_path, model_output_dir_path, signature_name):
    imported = tf.saved_model.load(saved_model_dir_path)
    f = imported.signatures[signature_name]
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(f, lower_control_flow=False)

    input_tensors = [tensor for tensor in frozen_func.inputs if tensor.dtype != tf.resource]
    output_tensors = frozen_func.outputs

    input_tensor_names = [tensor.name for tensor in frozen_func.inputs if tensor.dtype != tf.resource]
    output_tensor_names = [output.name for output in frozen_func.outputs]

    print('input_tensor_names:', input_tensor_names)
    print('output_tensor_names:', output_tensor_names)

    graph_def = run_graph_optimizations(
        graph_def,
        input_tensors,
        output_tensors,
        config=get_grappler_config(["constfold", "function"]),
        graph=frozen_func.graph)

    tf.io.write_graph(graph_or_graph_def=graph_def,
                        logdir=model_output_dir_path,
                        name='model_v2.pb',
                        as_text=False)
    
    print(f'Output model_v2.pb to {model_output_dir_path}/model_v2.pb')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model_dir_path', type=str, required=True, help='Input saved_model dir path')
    parser.add_argument('--model_output_dir_path', type=str, default='pb_from_saved_model', help='The output folder path of the converted model file (.pb)')
    parser.add_argument('--signature_name', type=str, default='serving_default', help='Signature name to be extracted from saved_model')
    args = parser.parse_args()

    saved_model_dir_path = args.saved_model_dir_path
    model_output_dir_path = args.model_output_dir_path
    signature_name = args.signature_name
    os.makedirs(model_output_dir_path, exist_ok=True)
    convert_saved_model_to_graph_def(saved_model_dir_path, model_output_dir_path, signature_name)

if __name__ == "__main__":
    main()