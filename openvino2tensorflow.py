'''
tensorflow==2.3.1

python3 openvino2tensorflow.py \
  --model_path=openvino/448x448/FP32/Resnet34_3inputs_448x448_20200609.xml \
  --output_saved_model=True \
  --output_pb=True \
  --output_weight_quant_tflite=True \
  --output_float16_quant_tflite=True \
  --output_no_quant_float32_tflite=True
'''

import os
import sys
import argparse
import struct
import numpy as np
import xml.etree.ElementTree as et
from openvino.inference_engine import IECore

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, ReLU, MaxPool2D, Reshape, Concatenate, Conv2DTranspose, Layer
from tensorflow.keras.initializers import Constant
from tensorflow.keras.backend import resize_images
from tensorflow.keras.activations import tanh, elu, sigmoid
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import sys
import tensorflow_datasets as tfds

def convert(model,
            model_output_path,
            output_saved_model,
            output_h5,
            output_pb,
            output_no_quant_float32_tflite,
            output_weight_quant_tflite,
            output_float16_quant_tflite):

    # for unpacking binary buffer
    format_config = { 'FP32': ['f', 4], 
                      'FP16': ['e', 2],
                      'I64' : ['q', 8],
                      'I32' : ['i', 4],
                      'I16' : ['h', 2],
                      'I8'  : ['b', 1],
                      'U8'  : ['B', 1]}

    # Read IR weight data
    with open(model+'.bin', 'rb') as f:
        binWeight = f.read()
    # Parse IR XML file, 
    tree = et.parse(model+'.xml')
    root = tree.getroot()
    edges = root.find('edges')
    layers = root.find('layers')
    tf_layers_dict = {}
    tf_edges = {}

    tf_inputs = []
    tf_outputs = []

    # edges
    for edge in edges:
        to_layer = int(edge.attrib['to-layer'])
        from_layer = int(edge.attrib['from-layer'])
        tf_edges.setdefault(to_layer, []).append(from_layer)

    # layers
    for idx, layer in enumerate(layers):
        layer_id = int(layer.attrib['id'])
        layer_name = layer.attrib['name'].replace('.', '_').replace('/', '_')
        data = layer.find('data')

        ### Parameter
        if layer.attrib['type'] == 'Parameter':
            if not data is None and 'shape' in data.attrib:
                shape_str  = data.attrib['shape'].split(',')
                shape = [int(s) for s in shape_str]
                tf_layers_dict[layer_id] = Input(shape=(shape[2], shape[3], shape[1]), batch_size=shape[0], name=layer_name)
                tf_inputs.append(tf_layers_dict[layer_id])

        ### Const
        elif layer.attrib['type'] == 'Const':
            if not data is None:
                if 'offset' in data.attrib and 'size' in data.attrib:
                    offset = int(data.attrib['offset'])
                    size   = int(data.attrib['size'])
                    shape_str  = data.attrib['shape'].split(',')
                    shape = [int(s) for s in shape_str]
                    blobBin = binWeight[offset:offset+size]
                    prec = layer.find('output').find('port').attrib['precision']
                    formatstring = '<' + format_config[prec][0] * (len(blobBin)//format_config[prec][1])
                    decodedwgt = np.array(list(struct.unpack(formatstring, blobBin))).reshape(shape)
                    tf_layers_dict[layer_id] = decodedwgt

        ### Convolution
        elif layer.attrib['type'] == 'Convolution':
            # port0 = [int(sdim.text) for sdim in layer.find('input')[0]]
            port1 = [int(sdim.text) for sdim in layer.find('input')[1]]
            filters = int(port1[0])
            kernel_size = [int(port1[2]), int(port1[3])]
            strides = [int(s) for s in data.attrib['strides'].split(',')]
            pads_begin = sum([int(s) for s in data.attrib['pads_begin'].split(',')])
            pads_end = sum([int(s) for s in data.attrib['pads_end'].split(',')])
            padding = ''
            if (pads_begin + pads_end) == 0:
                padding = 'valid'
            else:
                padding = 'same'
            dilations = [int(s) for s in data.attrib['dilations'].split(',')]
            tf_layers_dict[layer_id] = Conv2D(filters=filters,
                                              kernel_size=kernel_size,
                                              strides=strides,
                                              padding=padding,
                                              dilation_rate=dilations,
                                              use_bias=False,
                                              kernel_initializer=Constant(tf_layers_dict[tf_edges[layer_id][1]].transpose(2,3,1,0)))(tf_layers_dict[tf_edges[layer_id][0]])

        ### Add
        elif layer.attrib['type'] == 'Add':
            # 'Fused_Add_' == BiasAdd
            if 'Fused_Add_' in layer.attrib['name']:
                # Biasadd
                edge_id0 = tf_edges[layer_id][0]
                edge_id1 = tf_edges[layer_id][1]
                tf_layers_dict[layer_id] = tf.math.add(tf_layers_dict[edge_id0], tf_layers_dict[edge_id1].flatten())
            else:
                # Add
                tf_layers_dict[layer_id] = Add()([tf_layers_dict[from_layer_id].transpose(0,2,3,1) if type(tf_layers_dict[from_layer_id]) == np.ndarray else tf_layers_dict[from_layer_id] for from_layer_id in tf_edges[layer_id]])

        ### ReLU
        elif layer.attrib['type'] == 'ReLU':
            tf_layers_dict[layer_id] = ReLU()(tf_layers_dict[tf_edges[layer_id][0]])

        ### Tanh
        elif layer.attrib['type'] == 'Tanh':
            tf_layers_dict[layer_id] = tanh(tf_layers_dict[tf_edges[layer_id][0]])

        ### Elu
        elif layer.attrib['type'] == 'Elu':
            alpha = float(data.attrib['alpha'])
            tf_layers_dict[layer_id] = elu(tf_layers_dict[tf_edges[layer_id][0]], alpha=alpha)

        ### Sigmoid
        elif layer.attrib['type'] == 'Sigmoid':
            tf_layers_dict[layer_id] = sigmoid(tf_layers_dict[tf_edges[layer_id][0]])

        ### MaxPool
        elif layer.attrib['type'] == 'MaxPool':
            kernel_size =  [int(s) for s in data.attrib['kernel'].split(',')]
            strides = [int(s) for s in data.attrib['strides'].split(',')]
            pads_begin = sum([int(s) for s in data.attrib['pads_begin'].split(',')])
            pads_end = sum([int(s) for s in data.attrib['pads_end'].split(',')])
            padding = ''
            if (pads_begin + pads_end) == 0:
                padding = 'VALID'
            else:
                padding = 'SAME'
            tf_layers_dict[layer_id] = tf.nn.max_pool(tf_layers_dict[tf_edges[layer_id][0]], ksize=kernel_size, strides=strides, padding=padding)

        ### GroupConvolution
        elif layer.attrib['type'] == 'GroupConvolution':
            # port0 = [int(sdim.text) for sdim in layer.find('input')[0]]
            port1 = [int(sdim.text) for sdim in layer.find('input')[1]]
            depth_multiplier = int(port1[2])
            kernel_size = [int(port1[3]), int(port1[4])]
            strides = [int(s) for s in data.attrib['strides'].split(',')]
            pads_begin = sum([int(s) for s in data.attrib['pads_begin'].split(',')])
            pads_end = sum([int(s) for s in data.attrib['pads_end'].split(',')])
            padding = ''
            if (pads_begin + pads_end) == 0:
                padding = 'valid'
            else:
                padding = 'same'
            dilations = [int(s) for s in data.attrib['dilations'].split(',')]
            tf_layers_dict[layer_id] = DepthwiseConv2D(kernel_size=kernel_size,
                                                       strides=strides,
                                                       padding=padding,
                                                       depth_multiplier=depth_multiplier,
                                                       dilation_rate=dilations,
                                                       use_bias=False,
                                                       depthwise_initializer=Constant(tf_layers_dict[tf_edges[layer_id][1]].transpose(3,4,1,2,0)))(tf_layers_dict[tf_edges[layer_id][0]])

        ### ConvolutionBackpropData
        elif layer.attrib['type'] == 'ConvolutionBackpropData':
            # port0 = [int(sdim.text) for sdim in layer.find('input')[0]]
            port1 = [int(sdim.text) for sdim in layer.find('input')[1]]
            filters = int(port1[0])
            kernel_size = [int(port1[2]), int(port1[3])]
            strides = [int(s) for s in data.attrib['strides'].split(',')]
            pads_begin = sum([int(s) for s in data.attrib['pads_begin'].split(',')])
            pads_end = sum([int(s) for s in data.attrib['pads_end'].split(',')])
            padding = ''
            if (pads_begin + pads_end) == 0:
                padding = 'valid'
            else:
                padding = 'same'
            dilations = [int(s) for s in data.attrib['dilations'].split(',')]
            tf_layers_dict[layer_id] = Conv2DTranspose(filters=filters,
                                                       kernel_size=kernel_size,
                                                       strides=strides,
                                                       padding=padding,
                                                       dilation_rate=dilations,
                                                       use_bias=False,
                                                       kernel_initializer=Constant(tf_layers_dict[tf_edges[layer_id][1]].transpose(2,3,1,0)))(tf_layers_dict[tf_edges[layer_id][0]])

        ### Concat
        elif layer.attrib['type'] == 'Concat':
            tf_layers_dict[layer_id] = Concatenate()([tf_layers_dict[from_layer_id] for from_layer_id in tf_edges[layer_id]])

        ### Multiply
        elif layer.attrib['type'] == 'Multiply':
            tf_layers_dict[layer_id] = tf.math.multiply(tf_layers_dict[tf_edges[layer_id][0]], tf_layers_dict[tf_edges[layer_id][1]].transpose(0,2,3,1))

        ### Interpolate
        elif layer.attrib['type'] == 'Interpolate':
            mode = data.attrib['mode']
            in_port0 = [int(sdim.text) for sdim in layer.find('input')[0]]
            in_height = int(in_port0[2])
            in_width  = int(in_port0[3])
            out_port0 = [int(sdim.text) for sdim in layer.find('output')[0]]
            out_height = int(out_port0[2])
            out_width  = int(out_port0[3])

            h_scaling_factor = out_height // in_height
            w_scaling_factor  = out_width // in_width

            if mode == 'linear':
                tf_layers_dict[layer_id] = resize_images(tf_layers_dict[tf_edges[layer_id][0]], h_scaling_factor, w_scaling_factor, 'channels_last', interpolation='bilinear')
            elif mode == 'nearest':
                tf_layers_dict[layer_id] = resize_images(tf_layers_dict[tf_edges[layer_id][0]], h_scaling_factor, w_scaling_factor, 'channels_last', interpolation='nearest')
            else:
                print('The Interpolate - {} is not yet implemented.'.format(mode))
                sys.exit(-1)

        ### Result
        elif layer.attrib['type'] == 'Result':
            tf_layers_dict[layer_id] = tf.identity(tf_layers_dict[tf_edges[layer_id][0]], name=layer.attrib['name'].split('/')[0])
            tf_outputs.append(tf_layers_dict[layer_id])

        else:
            print('The {} layer is not yet implemented.'.format(layer.attrib['type']))
            sys.exit(-1)


    model = Model(inputs=tf_inputs, outputs=tf_outputs)
    model.summary()

    # saved_model output
    if output_saved_model:
        tf.saved_model.save(model, model_output_path)

    # .h5 output
    if output_h5:
        model.save('{}/model.h5'.format(model_output_path))

    # .pb output
    if output_pb:
        full_model = tf.function(lambda inputs: model(inputs))
        full_model = full_model.get_concrete_function(inputs=[tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in model.inputs])
        frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
        frozen_func.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                            logdir=".",
                            name='{}/model_float32.pb'.format(model_output_path),
                            as_text=False)

    # No Quantization - Input/Output=float32
    if output_no_quant_float32_tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open('{}/model_float32.tflite'.format(model_output_path), 'wb') as w:
            w.write(tflite_model)
        print("tflite convert complete! - {}/model_float32.tflite".format(model_output_path))

    # Weight Quantization - Input/Output=float32
    if output_weight_quant_tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        tflite_model = converter.convert()
        with open('{}/model_weight_quant.tflite'.format(model_output_path), 'wb') as w:
            w.write(tflite_model)
        print('Weight Quantization complete! - {}/model_weight_quant.tflite'.format(model_output_path))

    # Float16 Quantization - Input/Output=float32
    if output_float16_quant_tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_quant_model = converter.convert()
        with open('{}/model_float16_quant.tflite'.format(model_output_path), 'wb') as w:
            w.write(tflite_quant_model)
        print('Float16 Quantization complete! - {}/model_float16_quant.tflite'.format(model_output_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='input IR model path (.xml)')
    parser.add_argument('--model_output_path', type=str, default='saved_model', help='The output folder path of the converted model file')
    parser.add_argument('--output_saved_model', type=bool, default=True, help='saved_model output switch')
    parser.add_argument('--output_h5', type=bool, default=False, help='.h5 output switch')
    parser.add_argument('--output_pb', type=bool, default=False, help='.pb output switch')
    parser.add_argument('--output_no_quant_float32_tflite', type=bool, default=False, help='float32 tflite output switch')
    parser.add_argument('--output_weight_quant_tflite', type=bool, default=False, help='weight quant tflite output switch')
    parser.add_argument('--output_float16_quant_tflite', type=bool, default=False, help='float16 quant tflite output switch')
    args = parser.parse_args()
    model, ext = os.path.splitext(args.model_path)
    model_output_path = args.model_output_path.rstrip('/')
    if ext != '.xml':
        print('The specified model is not \'.xml\' file.')
        sys.exit(-1)
    output_saved_model = args.output_saved_model
    output_h5 = args.output_h5
    output_pb = args.output_pb
    output_no_quant_float32_tflite =  args.output_no_quant_float32_tflite
    output_weight_quant_tflite = args.output_weight_quant_tflite
    output_float16_quant_tflite = args.output_float16_quant_tflite
    if not output_saved_model and \
        not output_h5 and \
        not output_pb and \
        not output_no_quant_float32_tflite and \
        not output_weight_quant_tflite and \
        not output_float16_quant_tflite:
        print('Set at least one of the output switches (output_*) to true.')
        sys.exit(-1) 
    convert(model, model_output_path, output_saved_model, output_h5, output_pb,
            output_no_quant_float32_tflite, output_weight_quant_tflite, output_float16_quant_tflite)

if __name__ == "__main__":
    main()