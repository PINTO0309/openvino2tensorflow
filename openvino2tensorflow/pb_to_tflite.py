#! /usr/bin/env python
### tensorflow==2.3.1

import tensorflow.compat.v1 as tf
import shutil
import argparse
import os

def convert_graph_def_to_tflite(export_dir, graph_filepath, inputs, outputs):
    converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_filepath, inputs, outputs)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open("{}/model_from_pb_float32.tflite".format(export_dir), "wb").write(tflite_model)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pb_file_path', type=str, required=True, help='Input .pb file path (.pb)')
    parser.add_argument('--inputs', type=str, required=True, help='(e.g.1) input,input_1,input_2 / (e.g.2) images,input,param')
    parser.add_argument('--outputs', type=str, required=True, help='(e.g.1) output,output_1,output_2 / (e.g.2) Identity,Identity_1,output')
    parser.add_argument('--model_output_path', type=str, default='saved_model_from_pb', help='The output folder path of the converted model file')

    args = parser.parse_args()

    pb_file_path = args.pb_file_path
    inputs  = args.inputs.split(',')
    outputs = args.outputs.split(',')
    model_output_path = args.model_output_path

    shutil.rmtree(model_output_path, ignore_errors=True)
    os.makedirs(model_output_path, exist_ok=True)
    convert_graph_def_to_tflite(model_output_path, pb_file_path, inputs, outputs)

if __name__ == "__main__":
    main()