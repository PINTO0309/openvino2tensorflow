#! /usr/bin/env python
### tensorflow==2.3.1

import tensorflow.compat.v1 as tf
import shutil
import argparse
import os

def get_graph_def_from_file(graph_filepath):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        with tf.gfile.GFile(graph_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def

def convert_graph_def_to_saved_model(export_dir, graph_filepath, inputs, outputs):
    graph_def = get_graph_def_from_file(graph_filepath)
    with tf.Session(graph=tf.Graph()) as session:
        tf.import_graph_def(graph_def, name='')
        tf.saved_model.simple_save(
            session,
            export_dir,
            inputs= {t.rstrip(":0"):session.graph.get_tensor_by_name(t) for t in inputs},
            outputs={t.rstrip(":0"):session.graph.get_tensor_by_name(t) for t in outputs}
        )
        print('Optimized graph converted to SavedModel!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pb_file_path', type=str, required=True, help='Input .pb file path (.pb)')
    parser.add_argument('--inputs', type=str, required=True, help='(e.g.1) input:0,input:1,input:2 / (e.g.2) images:0,input:0,param:0')
    parser.add_argument('--outputs', type=str, required=True, help='(e.g.1) output:0,output:1,output:2 / (e.g.2) Identity:0,Identity:1,output:0')
    parser.add_argument('--model_output_path', type=str, default='saved_model_from_pb', help='The output folder path of the converted model file')

    args = parser.parse_args()

    pb_file_path = args.pb_file_path
    inputs  = args.inputs.split(',')
    outputs = args.outputs.split(',')
    model_output_path = args.model_output_path

    shutil.rmtree(model_output_path, ignore_errors=True)
    os.makedirs(model_output_path, exist_ok=True)
    convert_graph_def_to_saved_model(model_output_path, pb_file_path, inputs, outputs)

if __name__ == "__main__":
    main()
