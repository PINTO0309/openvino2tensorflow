#! /usr/bin/env python
'''
tensorflow==2.3.1

python3 openvino2tensorflow.py \
  --model_path=openvino/448x448/FP32/Resnet34_3inputs_448x448_20200609.xml \
  --output_saved_model=True \
  --output_pb=True \
  --output_weight_quant_tflite=True \
  --output_float16_quant_tflite=True \
  --output_no_quant_float32_tflite=True

python3 openvino2tensorflow.py \
  --model_path=openvino/kitti_192x640/FP32/footprints_kitti_192x640.xml \
  --output_saved_model=True \
  --output_no_quant_float32_tflite=True

python3 openvino2tensorflow.py \
  --model_path=openvino/dense_depth_nyu_480x640/FP32/dense_depth_nyu_480x640.xml \
  --output_saved_model=True \
  --output_no_quant_float32_tflite=True

python3 openvino2tensorflow.py \
  --model_path=openvino/480x640/FP32/u2netp_480x640.xml \
  --output_saved_model=True \
  --output_no_quant_float32_tflite=True

python3 openvino2tensorflow.py \
  --model_path=openvino/deeplabv3/FP32/deeplabv3.xml \
  --output_saved_model=True \
  --output_no_quant_float32_tflite=True

python3 openvino2tensorflow.py \
  --model_path=openvino/efficientnet-b0-pytorch/FP32/efficientnet-b0-pytorch.xml \
  --output_saved_model=True \
  --output_pb=True \
  --output_no_quant_float32_tflite=True \
  --debug \
  --debug_layer_number=5

python3 openvino2tensorflow.py \
  --model_path=openvino/mobilenet-v2-pytorch/FP32/mobilenet-v2-pytorch.xml \
  --output_saved_model=True \
  --output_no_quant_float32_tflite=True \
  --debug \
  --debug_layer_number=258

python3 openvino2tensorflow.py \
  --model_path=openvino/midasnet/FP32/midasnet.xml \
  --output_weight_and_json=True \
  --output_pb=True

python3 openvino2tensorflow.py \
  --model_path openvino/tf_efficientnet_lite3_256x256/FP32/tf_efficientnet_lite3.xml \
  --output_saved_model=True \
  --output_pb=True \
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
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, MaxPool2D, AveragePooling2D, Reshape, Conv2DTranspose, PReLU, Lambda
from tensorflow.keras.initializers import Constant
from tensorflow.keras.activations import elu, hard_sigmoid
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import sys
import tensorflow_datasets as tfds

def convert(model,
            model_output_path,
            output_saved_model,
            output_h5,
            output_weight_and_json,
            output_pb,
            output_no_quant_float32_tflite,
            output_weight_quant_tflite,
            output_float16_quant_tflite,
            replace_swish_and_hardswish,
            replace_prelu_and_minmax,
            debug,
            debug_layer_number):

    # for unpacking binary buffer
    format_config = { 'FP32': ['f', 4], 
                      'FP16': ['e', 2],
                      'I64' : ['q', 8],
                      'I32' : ['i', 4],
                      'I16' : ['h', 2],
                      'I8'  : ['b', 1],
                      'U8'  : ['B', 1]}

    # vino:    u8,    u16,    u32,    u64,   i8,   i16,   i32,   i64,     f16,     f32,              bf16, boolean
    # tf  : uint8, uint16, uint32, uint64, int8, int16, int32, int64, float16, float32, float64, bfloat16

    # type conversion table
    cast_type_ov_tf = { 'u8'  : tf.uint8,
                        'u16' : tf.uint16,
                        'u32' : tf.uint32,
                        'u64' : tf.uint64,
                        'i8'  : tf.int8,
                        'i16' : tf.int16,
                        'i32' : tf.int32,
                        'i64' : tf.int64,
                        'f16' : tf.float16,
                        'f32' : tf.float32,
                        'bf16': tf.bfloat16}

    # integer type table
    int_type_tf = [tf.uint8,
                   tf.uint16,
                   tf.uint32,
                   tf.uint64,
                   tf.int8,
                   tf.int16,
                   tf.int32,
                   tf.int64]

    # pad type conversion table
    pad_type_ov_tf = { 'constant' : 'CONSTANT',
                       'reflect'  : 'REFLECT',
                       'symmetric': 'SYMMETRIC',
                       'edge'     : 'REFLECT'}

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
    layer_id_port_dict = {}

    def get_num_of_outputs_per_layer_id(tf_edges):
        output_count_by_layer_id_tmp = {}
        for key in tf_edges.keys():
            key_tmp = key.split(':')[0]
            output_count_by_layer_id_tmp.setdefault(key_tmp, {'count' : 0, 'layer_id:port' : []})
            output_count_by_layer_id_tmp[key_tmp]['count'] += 1
            output_count_by_layer_id_tmp[key_tmp]['layer_id:port'].append(key)
        return output_count_by_layer_id_tmp

    def get_bere_layer_type(before_layer):
        t = type(tf_layers_dict[before_layer])
        if t == np.ndarray:
            # Const
            return 'const'
        else:
            try:
                return tf_layers_dict[before_layer.split(':')[0]].op.type
            except:
                # TopK
                return 'other'
 
    def get_tf_edges_from(tf_edges, layer_id, edge_index=-1):
        if edge_index == -1:
            # Add, Concat
            layer_list = []
            for edge_index in range(len(tf_edges[layer_id])):
                before_layer_type = get_bere_layer_type(tf_edges[layer_id][edge_index])
                if before_layer_type == 'Split':
                    layer_list.append(tf_edges[layer_id][edge_index])
                elif before_layer_type == 'other':
                    layer_list.append(tf_edges[layer_id][edge_index])
                else:
                    layer_list.append(tf_edges[layer_id][edge_index].split(':')[0])
            return layer_list
        else:
            # Other
            if layer_id in tf_edges:
                before_layer_type = get_bere_layer_type(tf_edges[layer_id][edge_index])
            else:
                for key in tf_edges.keys():
                    if layer_id in key:
                        before_layer_type = get_bere_layer_type(tf_edges[key][edge_index])
                        layer_id = key
                        break
            if before_layer_type == 'Split':
                return tf_edges[layer_id][edge_index]
            elif before_layer_type == 'other':
                return tf_edges[layer_id][edge_index]            
            else:
                return tf_edges[layer_id][edge_index].split(':')[0]

    # edges
    added_key_list = []
    for edge in edges:
        to_layer = edge.attrib['to-layer']
        from_layer = edge.attrib['from-layer']
        from_layer_port = edge.attrib['from-port']

        for layer in layers:
            if layer.attrib['id'] == to_layer:
                output_layer_ports = layer.find('output')
                if layer.attrib['type'] != 'Result' and len(output_layer_ports) >= 2:
                    for port in output_layer_ports:
                        tf_edges.setdefault('{}:{}'.format(to_layer, port.attrib['id']), []).append(from_layer)
                    added_key_list.append(to_layer)
                else:
                    tf_edges.setdefault(to_layer, [])

        for layer in layers:
            if layer.attrib['id'] == from_layer:
                output_layer_ports = layer.find('output')
                if len(output_layer_ports) >= 2:
                    tf_edges.setdefault(to_layer, []).append('{}:{}'.format(from_layer, from_layer_port))
                    if to_layer not in added_key_list:
                        added_key_list.append(to_layer)
                else:
                    if to_layer not in added_key_list:
                        tf_edges.setdefault(to_layer, []).append(from_layer)
                        added_key_list.append(to_layer)
                    else:
                        flg = 'not_found'
                        for key in tf_edges.keys():
                            if to_layer in key and from_layer in tf_edges[key]:
                                flg = 'found'
                                break
                        if flg == 'not_found':
                            tf_edges.setdefault(to_layer, []).append(from_layer)
                break
    del added_key_list

    layer_id_port_dict = get_num_of_outputs_per_layer_id(tf_edges)
    # print(layer_id_port_dict)

    # for i in tf_edges.items():
    #     print(i)
    # sys.exit(0)

    # layers
    for idx, layer in enumerate(layers):
        layer_id = layer.attrib['id']
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
                    shape_str = '1' if data.attrib['shape'] == '' else data.attrib['shape'].split(',')
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
                if 'auto_pad' in data.attrib:
                    if data.attrib['auto_pad'] == 'same_upper' or data.attrib['auto_pad'] == 'same_lower':
                        padding = 'same'
                    else:
                        padding = 'valid'
                else:
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
                                              kernel_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(2,3,1,0)))(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Add
        elif layer.attrib['type'] == 'Add':
            # 'Fused_Add_' == BiasAdd
            if len(tf_edges[layer_id]) == 2 and type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]) == np.ndarray:
                try:
                    # Biasadd
                    edge_id0 = get_tf_edges_from(tf_edges, layer_id, 0)
                    edge_id1 = get_tf_edges_from(tf_edges, layer_id, 1)
                    tf_layers_dict[layer_id] = tf.math.add(tf_layers_dict[edge_id0], tf_layers_dict[edge_id1].flatten())
                except:
                    # Add
                    edge_id0 = get_tf_edges_from(tf_edges, layer_id, 0)
                    edge_id1 = get_tf_edges_from(tf_edges, layer_id, 1)
                    tf_layers_dict[layer_id] = tf.math.add(tf_layers_dict[edge_id0], tf_layers_dict[edge_id1])
            else:
                # Add
                if len(get_tf_edges_from(tf_edges, layer_id)) == 2:
                    tmp_layers = [tf_layers_dict[from_layer_id].transpose(0,2,3,1) if type(tf_layers_dict[from_layer_id]) == np.ndarray else tf_layers_dict[from_layer_id] for from_layer_id in get_tf_edges_from(tf_edges, layer_id)]
                    tf_layers_dict[layer_id] = tf.math.add(tmp_layers[0], tmp_layers[1])
                else:
                    tf_layers_dict[layer_id] = tf.math.add_n([tf_layers_dict[from_layer_id].transpose(0,2,3,1) if type(tf_layers_dict[from_layer_id]) == np.ndarray else tf_layers_dict[from_layer_id] for from_layer_id in get_tf_edges_from(tf_edges, layer_id)])

        ### ReLU
        elif layer.attrib['type'] == 'ReLU':
            tf_layers_dict[layer_id] = tf.nn.relu(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### PReLU
        elif layer.attrib['type'] == 'PReLU':
            input_len = len(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape)
            alpha_len = len(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].shape)

            shared_axes = []
            if alpha_len == 1:
                shared_axes = [val + 1 for val in range(input_len - 1)]
            else:
                shared_axes = None

            if alpha_len == 4:
                if replace_prelu_and_minmax:
                    tf_layers_dict[layer_id] = tf.maximum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) + tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(0,2,3,1) * tf.minimum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                else:
                    tf_layers_dict[layer_id] = PReLU(alpha_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(0,2,3,1)), shared_axes=shared_axes)(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
            else:
                if replace_prelu_and_minmax:
                    tf_layers_dict[layer_id] = tf.maximum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) + tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)] * tf.minimum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                else:
                    tf_layers_dict[layer_id] = PReLU(alpha_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]), shared_axes=shared_axes)(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
    
        ### Clamp
        elif layer.attrib['type'] == 'Clamp':
            cmin = float(data.attrib['min'])
            cmax = float(data.attrib['max'])
            if cmin == 0.0 and cmax == 6.0:
                # ReLU6
                tf_layers_dict[layer_id] = tf.nn.relu6(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
            else:
                # Other
                tf_layers_dict[layer_id] = tf.clip_by_value(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], clip_value_min=cmin, clip_value_max=cmax)

        ### Tan
        elif layer.attrib['type'] == 'Tan':
            tf_layers_dict[layer_id] = tf.math.tan(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Tanh
        elif layer.attrib['type'] == 'Tanh':
            tf_layers_dict[layer_id] = tf.math.tanh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Elu
        elif layer.attrib['type'] == 'Elu':
            alpha = float(data.attrib['alpha'])
            tf_layers_dict[layer_id] = elu(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], alpha=alpha)

        ### HardSigmoid
        elif layer.attrib['type'] == 'HardSigmoid':
            tf_layers_dict[layer_id] = hard_sigmoid(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Sigmoid
        elif layer.attrib['type'] == 'Sigmoid':
            tf_layers_dict[layer_id] = tf.math.sigmoid(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Swish
        elif layer.attrib['type'] == 'Swish':
            if replace_swish_and_hardswish:
                # Hard-Swish
                tf_layers_dict[layer_id] = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] * tf.nn.relu6(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] + 3) * 0.16666667
            else:
                # Swish
                tf_layers_dict[layer_id] = tf.nn.swish(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### SoftPlus
        elif layer.attrib['type'] == 'SoftPlus':
            tf_layers_dict[layer_id] = tf.math.softplus(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### MaxPool
        elif layer.attrib['type'] == 'MaxPool':
            outport_size = sum([int(sdim.text) for sdim in layer.find('output')[0]])
            kernel_size =  [int(s) for s in data.attrib['kernel'].split(',')]
            strides = [int(s) for s in data.attrib['strides'].split(',')]
            pads_begin = sum([int(s) for s in data.attrib['pads_begin'].split(',')])
            pads_end = sum([int(s) for s in data.attrib['pads_end'].split(',')])
            padding = ''
            if (pads_begin + pads_end) == 0:
                if 'auto_pad' in data.attrib:
                    if data.attrib['auto_pad'] == 'same_upper' or data.attrib['auto_pad'] == 'same_lower':
                        padding = 'SAME'
                    else:
                        padding = 'VALID'
                else:
                    padding = 'VALID'
            else:
                padding = 'SAME'

            tf_layers_dict[layer_id] = tf.nn.max_pool(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], ksize=kernel_size, strides=strides, padding=padding)
            new_layer_outport_size = sum([sdim for sdim in tf_layers_dict[layer_id].shape])
            if outport_size != new_layer_outport_size:
                # Caffe -> TF
                tf_layers_dict[layer_id] = tf.nn.max_pool(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], ksize=kernel_size, strides=strides, padding='SAME')

        ### AvgPool
        elif layer.attrib['type'] == 'AvgPool':
            kernel_size =  [int(s) for s in data.attrib['kernel'].split(',')]
            strides = [int(s) for s in data.attrib['strides'].split(',')]
            # exclude_pad = data.attrib['exclude-pad']
            pads_begin = sum([int(s) for s in data.attrib['pads_begin'].split(',')])
            pads_end = sum([int(s) for s in data.attrib['pads_end'].split(',')])
            padding = ''
            if (pads_begin + pads_end) == 0:
                if 'auto_pad' in data.attrib:
                    if data.attrib['auto_pad'] == 'same_upper' or data.attrib['auto_pad'] == 'same_lower':
                        padding = 'same'
                    else:
                        padding = 'valid'
                else:
                    padding = 'valid'
            else:
                padding = 'same'
            tf_layers_dict[layer_id] = AveragePooling2D(pool_size=kernel_size, strides=strides, padding=padding)(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### GroupConvolution
        elif layer.attrib['type'] == 'GroupConvolution':
            port0 = [int(sdim.text) for sdim in layer.find('input')[0]]
            port1 = [int(sdim.text) for sdim in layer.find('input')[1]]
            depth_multiplier = 1
            kernel_size = [int(port1[3]), int(port1[4])]
            strides = [int(s) for s in data.attrib['strides'].split(',')]
            pads_begin = sum([int(s) for s in data.attrib['pads_begin'].split(',')])
            pads_end = sum([int(s) for s in data.attrib['pads_end'].split(',')])
            padding = ''
            if (pads_begin + pads_end) == 0:
                if 'auto_pad' in data.attrib:
                    if data.attrib['auto_pad'] == 'same_upper' or data.attrib['auto_pad'] == 'same_lower':
                        padding = 'same'
                    else:
                        padding = 'valid'
                else:
                    padding = 'valid'
            else:
                padding = 'same'
            dilations = [int(s) for s in data.attrib['dilations'].split(',')]

            if int(port1[1]) > 1:
                # Conv2D with groups
                filters = int(port0[1])
                groups = int(port1[0])

                convs = []
                kernel = None
                if len(port1) == 5:
                    kernel = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(3,4,2,1,0)
                    for i in range(groups):
                        convs.append(Conv2D(filters=filters // groups,
                                            kernel_size=kernel_size,
                                            strides=strides,
                                            padding=padding,
                                            dilation_rate=dilations,
                                            use_bias=False,
                                            kernel_initializer=Constant(kernel[:,:,:,:,i])))
                else:
                    kernel = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(2,3,1,0)
                    for i in range(groups):
                        convs.append(Conv2D(filters=filters // groups,
                                            kernel_size=kernel_size,
                                            strides=strides,
                                            padding=padding,
                                            dilation_rate=dilations,
                                            use_bias=False,
                                            kernel_initializer=Constant(kernel[:,:,:,i])))
 
                x_splits = tf.split(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], groups, -1)
                x_outputs = [conv(x_split) for x_split, conv in zip(x_splits, convs)]
                tf_layers_dict[layer_id] = tf.concat(x_outputs, -1)
       
            else:
                # DepthwiseConv2D
                tf_layers_dict[layer_id] = DepthwiseConv2D(kernel_size=kernel_size,
                                                          strides=strides,
                                                          padding=padding,
                                                          depth_multiplier=depth_multiplier,
                                                          dilation_rate=dilations,
                                                          use_bias=False,
                                                          depthwise_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(3,4,1,2,0)))(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### ConvolutionBackpropData
        elif layer.attrib['type'] == 'ConvolutionBackpropData':
            # port0 = [int(sdim.text) for sdim in layer.find('input')[0]]
            port1 = [int(sdim.text) for sdim in layer.find('input')[1]]
            port2 = [int(sdim.text) for sdim in layer.find('output')[0]]
            filters = int(port2[1])
            kernel_size = [int(port1[2]), int(port1[3])]
            strides = [int(s) for s in data.attrib['strides'].split(',')]
            pads_begin = sum([int(s) for s in data.attrib['pads_begin'].split(',')])
            pads_end = sum([int(s) for s in data.attrib['pads_end'].split(',')])
            padding = ''
            if (pads_begin + pads_end) == 0:
                if 'auto_pad' in data.attrib:
                    if data.attrib['auto_pad'] == 'same_upper' or data.attrib['auto_pad'] == 'same_lower':
                        padding = 'same'
                    else:
                        padding = 'valid'
                else:
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
                                                       kernel_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(2,3,1,0)))(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Concat
        elif layer.attrib['type'] == 'Concat':
            axis = -1
            if 'axis' in data.attrib:
                axis = int(data.attrib['axis'])
            if axis == 1 and len(np.asarray(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape)) == 4:
                axis = -1
            tf_layers_dict[layer_id] = tf.concat([tf_layers_dict[from_layer_id] for from_layer_id in get_tf_edges_from(tf_edges, layer_id)], axis=axis)

        ### Multiply
        elif layer.attrib['type'] == 'Multiply':
            if len(tf_edges[layer_id]) == 2 and (type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]) == np.ndarray):
                if tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].ndim == 4:
                    # 4D - NCHW->NHWC
                    tf_layers_dict[layer_id] = tf.math.multiply(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(0,2,3,1).astype(np.float32))
                else:
                    # unknown
                    tf_layers_dict[layer_id] = tf.math.multiply(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])

            elif len(tf_edges[layer_id]) == 2 and (type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) == np.ndarray):
                if tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].ndim == 4:
                    # 4D - NCHW->NHWC
                    tf_layers_dict[layer_id] = tf.math.multiply(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].transpose(0,2,3,1).astype(np.float32), tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])
                else:
                    # unknown
                    tf_layers_dict[layer_id] = tf.math.multiply(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])

            else:
                # unknown
                tf_layers_dict[layer_id] = tf.math.multiply(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])

        ### Interpolate
        elif layer.attrib['type'] == 'Interpolate':
            mode = data.attrib['mode']
            antialias = False if int(data.attrib['antialias']) == 0 else True
            out_port0 = [int(sdim.text) for sdim in layer.find('output')[0]]
            out_height = int(out_port0[2])
            out_width  = int(out_port0[3])

            input_shape = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape
            input_shape_height = input_shape[1]
            input_shape_width  = input_shape[2]
            upsampling_factor_height = out_height // input_shape_height
            upsampling_factor_width  = out_width // input_shape_width

            def upsampling2d_bilinear(x, upsampling_factor_height, upsampling_factor_width):
                h = x.shape[2] * upsampling_factor_width
                w = x.shape[1] * upsampling_factor_height
                return tf.compat.v1.image.resize_bilinear(x, (h, w))

            def upsampling2d_nearest(x, upsampling_factor_height, upsampling_factor_width):
                h = x.shape[2] * upsampling_factor_width
                w = x.shape[1] * upsampling_factor_height
                return tf.compat.v1.image.resize_nearest_neighbor(x, (h, w))

            if (upsampling_factor_height * input_shape_height) == out_height and (upsampling_factor_width * input_shape_width) == out_width and upsampling_factor_height >= 1.0 and upsampling_factor_width >= 1.0:
                # Upsampling
                if mode == 'linear':
                    tf_layers_dict[layer_id] = Lambda(upsampling2d_bilinear,
                                                        arguments={'upsampling_factor_height': upsampling_factor_height,
                                                                   'upsampling_factor_width': upsampling_factor_width})(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                elif mode == 'nearest':
                    tf_layers_dict[layer_id] = Lambda(upsampling2d_nearest,
                                                        arguments={'upsampling_factor_height': upsampling_factor_height,
                                                                   'upsampling_factor_width': upsampling_factor_width})(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                else:
                    print('The Interpolate - {} is not yet implemented.'.format(mode))
                    sys.exit(-1)
            else:
                # Others
                if mode == 'linear':
                    tf_layers_dict[layer_id] = tf.image.resize(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], [out_height, out_width], method='bilinear', antialias=antialias)
                elif mode == 'nearest':
                    tf_layers_dict[layer_id] = tf.image.resize(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], [out_height, out_width], method='nearest', antialias=antialias)
                else:
                    print('The Interpolate - {} is not yet implemented.'.format(mode))
                    sys.exit(-1)

        ### ShapeOf
        elif layer.attrib['type'] == 'ShapeOf':
            tf_layers_dict[layer_id] = tf.shape(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], out_type=tf.int64)

        ### Convert
        elif layer.attrib['type'] == 'Convert':
            # vino:    u8,    u16,    u32,    u64,   i8,   i16,   i32,   i64,     f16,     f32,              bf16, boolean
            # tf  : uint8, uint16, uint32, uint64, int8, int16, int32, int64, float16, float32, float64, bfloat16
            destination_type = data.attrib['destination_type']
            tf_layers_dict[layer_id] = tf.cast(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], cast_type_ov_tf[destination_type])

        ### StridedSlice - TODO
        elif layer.attrib['type'] == 'StridedSlice':
            begin_mask       = data.attrib['begin_mask']
            end_mask         = data.attrib['end_mask']
            ellipsis_mask    = data.attrib['ellipsis_mask']
            new_axis_mask    = data.attrib['new_axis_mask']
            shrink_axis_mask = data.attrib['shrink_axis_mask']

            # begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask
            begin_mask       = np.asarray([int(val) for val in begin_mask.split(',')])
            end_mask         = np.asarray([int(val) for val in end_mask.split(',')])
            ellipsis_mask    = np.asarray([int(val) for val in ellipsis_mask.split(',')])
            new_axis_mask    = np.asarray([int(val) for val in new_axis_mask.split(',')])
            shrink_axis_mask = np.asarray([int(val) for val in shrink_axis_mask.split(',')])

            if type(begin_mask) == np.ndarray and len(begin_mask) == 4:
                begin_mask[0], begin_mask[1], begin_mask[2], begin_mask[3] = begin_mask[0], begin_mask[2], begin_mask[3], begin_mask[1]
            if np.sum(begin_mask) == len(begin_mask):
                begin_mask = -1
            else:
                begin_mask = np.argmin(begin_mask)

            if type(end_mask) == np.ndarray and len(end_mask) == 4:
                end_mask[0], end_mask[1], end_mask[2], end_mask[3] = end_mask[0], end_mask[2], end_mask[3], end_mask[1]
            if np.sum(end_mask) == len(end_mask):
                end_mask = -1
            else:
                end_mask = np.argmin(end_mask)

            if type(ellipsis_mask) == np.ndarray and len(ellipsis_mask) == 4:
                ellipsis_mask[0], ellipsis_mask[1], ellipsis_mask[2], ellipsis_mask[3] = ellipsis_mask[0], ellipsis_mask[2], ellipsis_mask[3], ellipsis_mask[1]
            ellipsis_mask = np.argmin(ellipsis_mask)

            if type(new_axis_mask) == np.ndarray and len(new_axis_mask) == 4:
                new_axis_mask[0], new_axis_mask[1], new_axis_mask[2], new_axis_mask[3] = new_axis_mask[0], new_axis_mask[2], new_axis_mask[3], new_axis_mask[1]
            new_axis_mask = np.argmin(new_axis_mask)
            

            if type(shrink_axis_mask) == np.ndarray and len(shrink_axis_mask) == 4:
                shrink_axis_mask[0], shrink_axis_mask[1], shrink_axis_mask[2], shrink_axis_mask[3] = shrink_axis_mask[0], shrink_axis_mask[2], shrink_axis_mask[3], shrink_axis_mask[1]
            shrink_axis_mask = np.argmin(shrink_axis_mask)

            # begin, end, strides
            begin   = np.asarray([int(val) for val in tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]])
            end     = np.asarray([int(val) for val in tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]])
            strides = np.asarray([int(val) for val in tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 3)]])

            if len(begin) == 4:
                begin[0], begin[1], begin[2], begin[3] = begin[0], begin[2], begin[3], begin[1]

            if len(end) == 4:
                end[0], end[1], end[2], end[3] = end[0], end[2], end[3], end[1]

            for idx, (b, e) in enumerate(zip(begin, end)):
                if b == 0 and b == e:
                    begin[idx] = 0
                    end[idx] = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape[idx]

            if len(strides) == 4:
                strides[0], strides[1], strides[2], strides[3] = strides[0], strides[2], strides[3], strides[1]

            tf_layers_dict[layer_id] = tf.strided_slice(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                                        begin=begin,
                                                        end=end,
                                                        strides=strides,
                                                        begin_mask=begin_mask,
                                                        end_mask=end_mask,
                                                        ellipsis_mask=ellipsis_mask,
                                                        new_axis_mask=new_axis_mask,
                                                        shrink_axis_mask=shrink_axis_mask)

        ### Pad
        elif layer.attrib['type'] == 'Pad':
            pad_mode = pad_type_ov_tf[data.attrib['pad_mode']]
            pads_begin = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)] # [0,0,1,1]
            pads_end   = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)] # [0,0,1,1]

            if (pads_end[0] == 0 and pads_begin[0] == 0):
               pad_b_bottom = pad_b_top = 0
            else:
               pad_b_top = pads_begin[0]
               pad_b_bottom = pads_end[0]

            if (pads_end[1] == 0 and pads_begin[1] == 0):
               pad_c_bottom = pad_c_top = 0
            else:
               pad_c_top = pads_begin[1]
               pad_c_bottom = pads_end[1]

            if (pads_end[2] == 0 and pads_begin[2] == 0):
               pad_bottom = pad_top = 0
            else:
               pad_top = pads_begin[2]
               pad_bottom = pads_end[2]

            if (pads_end[3] == 0 and pads_begin[3] == 0):
               pad_right = pad_left = 0
            else:
               pad_left = pads_begin[3]
               pad_right = pads_end[3]

            paddings = [[pad_b_top, pad_b_bottom], [pad_top, pad_bottom], [pad_left, pad_right], [pad_c_top, pad_c_bottom]]

            pad_value  = np.float32(0.0)
            if 'pad_value' in data.attrib:
                pad_value = np.float32(data.attrib['pad_value'])

            tf_layers_dict[layer_id] = tf.pad(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], paddings, mode=pad_mode, constant_values=pad_value)

        ### TopK
        elif layer.attrib['type'] == 'TopK':
            # axis = int(data.attrib['axis'])
            # index_element_type = data.attrib['index_element_type']
            # mode = data.attrib['mode']
            # sort = data.attrib['sort']
            layer_id_values  = layer_id_port_dict[layer_id]['layer_id:port'][0]
            layer_id_indices = layer_id_port_dict[layer_id]['layer_id:port'][1]
            try:
                tf_layers_dict[layer_id_values], tf_layers_dict[layer_id_indices] = tf.math.top_k(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], k=int(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]), sorted=True)
            except:
                tf_layers_dict[layer_id_values], tf_layers_dict[layer_id_indices] = tf.math.top_k(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], k=int(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)][0]), sorted=True)

        ### Transpose
        elif layer.attrib['type'] == 'Transpose':
            input_shape_len = len(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape)
            temp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
            perm = []
            if input_shape_len == 4:
                if type(temp) == np.ndarray:
                    for idx, dim in enumerate(temp):
                        if dim == 0:
                            perm.append(0)
                        elif dim == 1:
                            perm.append(input_shape_len - 1)
                        else:
                            perm.append(dim - 1)
                else:
                    # TODO
                    shape = tf.shape(temp)
                    for idx, dim in enumerate(shape):
                        if dim == 0:
                            perm.append(0)
                        elif dim == 1:
                            perm.append(input_shape_len - 1)
                        else:
                            perm.append(dim - 1)
            elif input_shape_len == 5:
                perm_tmp = ''
                for idx, dim in enumerate(temp):
                    perm_tmp += str(dim)
                if perm_tmp == '02134':
                    perm.append(0)
                    perm.append(1)
                    perm.append(2)
                    perm.append(4)
                    perm.append(3)
                else:
                    # TODO
                    for idx, dim in enumerate(temp):
                        perm.append(dim)
                    # tf_layers_dict[layer_id] = tf.identity(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                    # continue
            else:
                for idx, dim in enumerate(temp):
                    perm.append(dim)
            tf_layers_dict[layer_id] = tf.transpose(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], perm=perm)

        ### Squeeze
        elif layer.attrib['type'] == 'Squeeze':
            axis = None
            if len(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]) == 1:
                axis = int(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])
                if axis == 1:
                    axis = -1
                elif axis >= 2:
                    axis -= 1
            else:
                for idx, part_axis in enumerate(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]):
                    if part_axis == 1:
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)][idx] = -1
                    elif part_axis >= 2:
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)][idx] -= 1
                axis = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
            try:
                tf_layers_dict[layer_id] = tf.squeeze(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], axis=axis)
            except:
                tf_layers_dict[layer_id] = tf.squeeze(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], axis=-1)

        ### Gather
        elif layer.attrib['type'] == 'Gather':
            axis = int(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)])
            temp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
            input_shape = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape[0]
            indices = []
            if type(temp) == np.ndarray:
                # for idx, dim in enumerate(temp):
                #     if idx == 0:
                #         indices.append(0)
                #     elif idx == input_shape - 1:
                #         indices.append(1)
                #     else:
                #         indices.append(dim + 1)
                for idx, dim in enumerate(temp):
                    indices.append(dim)
            else:
                # TODO
                shape = tf.shape(temp)
                for idx, dim in enumerate(shape):
                    if idx == 0:
                        indices.append(0)
                    elif idx == input_shape - 1:
                        indices.append(1)
                    else:
                        indices.append(dim + 1)
            tf_layers_dict[layer_id] = tf.gather(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], indices, axis=axis)

        ### ReduceMean, ReduceMax, ReduceMin, ReduceSum, ReduceProd - TODO
        elif layer.attrib['type'] == 'ReduceMean' or layer.attrib['type'] == 'ReduceMax' or layer.attrib['type'] == 'ReduceMin' or layer.attrib['type'] == 'ReduceSum' or layer.attrib['type'] == 'ReduceProd':
            keep_dims = True if data.attrib['keep_dims'] == "True" else False
            axis = None
            if type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]) == np.ndarray and len(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]) == 1:
                axis = int(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])
                if axis == 1:
                    axis = -1
                elif axis >= 2:
                    axis -= 1
            elif type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]) != np.ndarray and len(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].shape) == 1:
                axis = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)] - 1
            else:
                for idx, part_axis in enumerate(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]):
                    if part_axis == 1:
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)][idx] = -1
                    elif part_axis >= 2:
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)][idx] -= 1
                axis = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
            if layer.attrib['type'] == 'ReduceMean':
                tf_layers_dict[layer_id] = tf.math.reduce_mean(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], axis=axis, keepdims=keep_dims)
            elif layer.attrib['type'] == 'ReduceMax':
                tf_layers_dict[layer_id] = tf.math.reduce_max(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], axis=axis, keepdims=keep_dims)
            elif layer.attrib['type'] == 'ReduceMin':
                tf_layers_dict[layer_id] = tf.math.reduce_min(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], axis=axis, keepdims=keep_dims)
            elif layer.attrib['type'] == 'ReduceSum':
                tf_layers_dict[layer_id] = tf.math.reduce_sum(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], axis=axis, keepdims=keep_dims)
            elif layer.attrib['type'] == 'ReduceProd':
                tf_layers_dict[layer_id] = tf.math.reduce_prod(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], axis=axis, keepdims=keep_dims)

        ### MatMul
        elif layer.attrib['type'] == 'MatMul':
            if not data is None and 'a' in data.attrib:
                transpose_a = True if int(data.attrib['a']) == 1 else False
            if not data is None and 'b' in data.attrib:
                transpose_b = True if int(data.attrib['b']) == 1 else False
            if not data is None and 'transpose_a' in data.attrib:
                try:
                    transpose_a = True if int(data.attrib['transpose_a']) == 1 else False
                except:
                    transpose_a = True if data.attrib['transpose_a'] == 'True' else False
            if not data is None and 'transpose_b' in data.attrib:
                try:
                    transpose_b = True if int(data.attrib['transpose_b']) == 1 else False
                except:
                    transpose_b = True if data.attrib['transpose_b'] == 'True' else False
            tf_layers_dict[layer_id] = tf.linalg.matmul(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                                                        transpose_a, transpose_b)

        ### Reshape - TODO
        elif layer.attrib['type'] == 'Reshape':
            op1 = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
            op2 = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
            op_type1 = type(op1)
            op_type2 = type(op2)
            op_len1 = len(np.asarray(op1.shape))
            op_len2 = len(np.asarray(op2.shape))

            shape = []
            if op_type1 != np.ndarray and op_type2 != np.ndarray:
                # op and op
                # print('@@@@@@@@@@@@@@@@@ op / const - route0', op_len1, op_len2, op1, op2)
                if op_len2 > 1:
                    shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]

                elif op_len2 == 1 and op2.shape[0] == 2:
                    # TODO
                    shape = op2

                elif op_len2 == 1 and op2.shape[0] == 3:
                    # TODO
                    shape = op2

                elif op_len2 == 1 and op2.shape[0] == 4:
                    # TODO
                    shape = op2

                elif op_len2 == 1 and op2.shape[0] == 5:
                    # TODO
                    # shape_tmp = []
                    # shape_tmp.append(op2[0])
                    # shape_tmp.append(op2[1])
                    # shape_tmp.append(op2[2])
                    # shape_tmp.append(op2[3])
                    # shape_tmp.append(op2[4])
                    shape = op2

                elif op_len2 == 1 and op2.shape[0] == 6:
                    # YoloV4
                    # shape_tmp = []
                    # shape_tmp.append(op2[0])
                    # shape_tmp.append(op2[1])
                    # shape_tmp.append(op2[2])
                    # shape_tmp.append(op2[3])
                    # shape_tmp.append(op2[4])
                    # shape_tmp.append(op2[5])
                    shape = op2

            elif op_type1 != np.ndarray and op_type2 == np.ndarray:
                # op and const
                if op_len2 == 4:
                    # print('@@@@@@@@@@@@@@@@@ op / const - route1', op_len2)
                    op2 = op2.transpose(0,2,3,1)
                    shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]
                elif op_len2 > 4:
                    # print('@@@@@@@@@@@@@@@@@ op / const - route2', op_len2)
                    shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]
                elif op_len2 == 1 and op2.shape[0] == 2:
                    shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]
                elif op_len2 == 1 and op2.shape[0] == 3:
                    shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]
                    if op_len1 > len(op2):
                        if shape[2] == -1:
                            shape[0], shape[1], shape[2] = shape[0], shape[2], shape[1]

                elif op_len2 == 1 and op2.shape[0] == 4:
                    # print('@@@@@@@@@@@@@@@@@ op / const - route3', op_len2, shape)
                    shape_tmp = []
                    shape_tmp.append(op2[0])
                    shape_tmp.append(op2[2])
                    shape_tmp.append(op2[3])
                    shape_tmp.append(op2[1])
                    shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(shape_tmp)]
                elif op_len2 == 1 and op2.shape[0] == 5:
                    # print('@@@@@@@@@@@@@@@@@ op / const - route4', op_len2, shape)
                    shape_tmp = []
                    if op2[1] == op2[2]:
                        shape_tmp.append(op2[0])
                        shape_tmp.append(op2[1])
                        shape_tmp.append(op2[2])
                        shape_tmp.append(op2[3])
                        shape_tmp.append(op2[4])
                    else:
                        shape_tmp.append(op2[0])
                        shape_tmp.append(op2[3])
                        shape_tmp.append(op2[4])
                        shape_tmp.append(op2[1])
                        shape_tmp.append(op2[2])

                    shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(shape_tmp)]
                elif op_len2 == 1 and op2.shape[0] == 6:
                    # print('@@@@@@@@@@@@@@@@@ op / const - route5', op_len2)
                    # YoloV4
                    shape_tmp = []
                    shape_tmp.append(op2[0])
                    shape_tmp.append(op2[2])
                    shape_tmp.append(op2[3])
                    shape_tmp.append(op2[4])
                    shape_tmp.append(op2[5])
                    shape_tmp.append(op2[1])
                    shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(shape_tmp)]
                else:
                    # print('@@@@@@@@@@@@@@@@@ op / const - route6', op1, op2, op_len2, np.asarray(op2.shape))
                    for i in range(op2.shape[0]):
                        shape.append(op2[i])

            elif op_type1 == np.ndarray and op_type2 != np.ndarray:
                # const and op
                if op_len1 == 4:
                    op1 = op1.transpose(0,2,3,1)
                    shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]
                elif op_len1 > 4:
                    shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]
                # elif op_len1 == 1 and op1.shape[0] == 5:
                #     shape_tmp = []
                #     shape_tmp.append(op2[0])
                #     shape_tmp.append(op2[3])
                #     shape_tmp.append(op2[4])
                #     shape_tmp.append(op2[1])
                #     shape_tmp.append(op2[2])
                # elif op_len2 == 1 and op2.shape[0] == 6:
                #     # YoloV4
                #     shape_tmp = []
                #     shape_tmp.append(op2[0])
                #     shape_tmp.append(op2[2])
                #     shape_tmp.append(op2[3])
                #     shape_tmp.append(op2[4])
                #     shape_tmp.append(op2[5])
                #     shape_tmp.append(op2[1])
                else:
                    for i in range(op2.shape[0]):
                        shape.append(op2[i])           
            else:
                # const and const
                if op_len1 == 4:
                    op1 = op1.transpose(0,2,3,1)
                if op_len2 == 4:
                    op2 = op2.transpose(0,2,3,1)
                shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]

            # print('@@@@@@@@@@@@@@@@@ op / const', op1, op2, shape)
            tf_layers_dict[layer_id] = tf.reshape(op1, shape)

        ### Range - TODO
        elif layer.attrib['type'] == 'Range':
            dtype = cast_type_ov_tf[data.attrib['output_type']]
            tf_layers_dict[layer_id] = tf.range(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)][0],
                                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                                                delta=int(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]),
                                                dtype=dtype)

        ### Exp
        elif layer.attrib['type'] == 'Exp':
            tf_layers_dict[layer_id] = tf.math.exp(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Abs
        elif layer.attrib['type'] == 'Abs':
            tf_layers_dict[layer_id] = tf.math.abs(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### SoftMax
        elif layer.attrib['type'] == 'SoftMax':
            axis = int(data.attrib['axis'])
            if axis == 1 and len(np.asarray(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape)) == 4:
                axis = -1
            tf_layers_dict[layer_id] = tf.nn.softmax(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], axis=axis)

        ### Negative
        elif layer.attrib['type'] == 'Negative':
            tf_layers_dict[layer_id] = tf.math.negative(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Maximum
        elif layer.attrib['type'] == 'Maximum':
            # No broadcast
            tf_layers_dict[layer_id] = tf.math.maximum(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])

        ### Minimum
        elif layer.attrib['type'] == 'Minimum':
            # No broadcast
            tf_layers_dict[layer_id] = tf.math.minimum(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])

        ### Acos
        elif layer.attrib['type'] == 'Acos':
            tf_layers_dict[layer_id] = tf.math.acos(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Acosh
        elif layer.attrib['type'] == 'Acosh':
            tf_layers_dict[layer_id] = tf.math.acosh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Asin
        elif layer.attrib['type'] == 'Asin':
            tf_layers_dict[layer_id] = tf.math.asin(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Asinh
        elif layer.attrib['type'] == 'Asinh':
            tf_layers_dict[layer_id] = tf.math.asinh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Atan
        elif layer.attrib['type'] == 'Atan':
            tf_layers_dict[layer_id] = tf.math.atan(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Atanh
        elif layer.attrib['type'] == 'Atanh':
            tf_layers_dict[layer_id] = tf.math.atanh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Ceiling
        elif layer.attrib['type'] == 'Ceiling':
            tf_layers_dict[layer_id] = tf.math.ceil(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Cos
        elif layer.attrib['type'] == 'Cos':
            tf_layers_dict[layer_id] = tf.math.cos(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Cosh
        elif layer.attrib['type'] == 'Cosh':
            tf_layers_dict[layer_id] = tf.math.cosh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Sin
        elif layer.attrib['type'] == 'Sin':
            tf_layers_dict[layer_id] = tf.math.sin(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Sinh
        elif layer.attrib['type'] == 'Sinh':
            tf_layers_dict[layer_id] = tf.math.sinh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Divide
        elif layer.attrib['type'] == 'Divide':
            x_np_type = None
            y_np_type = None

            x = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
            if type(x) == np.ndarray and x.dtype == np.int:
                x_np_type = tf.int32
            elif type(x) == np.ndarray:
                x_np_type = tf.float32
            else:
                x_np_type = x.dtype

            y = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
            if type(y) == np.ndarray and y.dtype == np.int:
                y_np_type = tf.int32
            elif type(y) == np.ndarray:
                y_np_type = tf.float32
            else:
                y_np_type = y.dtype

            if (x_np_type in int_type_tf) and (y_np_type in int_type_tf):
                # floordiv
                tf_layers_dict[layer_id] = tf.math.floordiv(x, y)
            else:
                # divide
                tf_layers_dict[layer_id] = tf.math.divide(x, y)

        ### Erf
        elif layer.attrib['type'] == 'Erf':
            tf_layers_dict[layer_id] = tf.math.erf(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Floor
        elif layer.attrib['type'] == 'Floor':
            tf_layers_dict[layer_id] = tf.math.floor(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### FloorMod
        elif layer.attrib['type'] == 'FloorMod':
            tf_layers_dict[layer_id] = tf.math.floormod(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])

        ### HSwish
        elif layer.attrib['type'] == 'HSwish':
            if replace_swish_and_hardswish:
                # Swish
                tf_layers_dict[layer_id] = tf.nn.swish(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
            else:
                # Hard-Swish
                tf_layers_dict[layer_id] = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] * tf.nn.relu6(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] + 3) * 0.16666667

        ### Log
        elif layer.attrib['type'] == 'Log':
            tf_layers_dict[layer_id] = tf.math.log(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Power
        elif layer.attrib['type'] == 'Power':
            # No broadcast
            tf_layers_dict[layer_id] = tf.math.pow(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])

        ### Mish
        elif layer.attrib['type'] == 'Mish':
            tf_layers_dict[layer_id] = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] * tf.math.tanh(tf.math.softplus(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]))

        ### Selu
        elif layer.attrib['type'] == 'Selu':
            tf_layers_dict[layer_id] = tf.nn.selu(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### Subtract
        elif layer.attrib['type'] == 'Subtract':
            # No broadcast
            tf_layers_dict[layer_id] = tf.math.subtract(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])
    
        ### Unsqueeze - TODO
        elif layer.attrib['type'] == 'Unsqueeze':
            input_shape = np.asarray(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape)
            indices = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]

            if len(input_shape) == 1 and indices == [0]:
                tf_layers_dict[layer_id] = tf.identity(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
            elif len(input_shape) > 1 and len(indices) > 1:
                print('The multi-dimensional indices specification in Unsqueeze is not yet implemented.')
                sys.exit(-1)
            else:
                tf_layers_dict[layer_id] = tf.expand_dims(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], indices[0])

        ### Equal
        elif layer.attrib['type'] == 'Equal':
            tf_layers_dict[layer_id] = tf.math.equal(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])

        ### NotEqual
        elif layer.attrib['type'] == 'NotEqual':
            tf_layers_dict[layer_id] = tf.math.not_equal(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])

        ### Greater
        elif layer.attrib['type'] == 'Greater':
            tf_layers_dict[layer_id] = tf.math.greater(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])

        ### GreaterEqual
        elif layer.attrib['type'] == 'GreaterEqual':
            tf_layers_dict[layer_id] = tf.math.greater_equal(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])

        ### Less
        elif layer.attrib['type'] == 'Less':
            tf_layers_dict[layer_id] = tf.math.less(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])

        ### LessEqual
        elif layer.attrib['type'] == 'LessEqual':
            tf_layers_dict[layer_id] = tf.math.less_equal(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])

        ### Select
        elif layer.attrib['type'] == 'Select':
            tf_layers_dict[layer_id] = tf.raw_ops.SelectV2(condition=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                                           t=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                                                           e=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)])

        ### LogicalAnd
        elif layer.attrib['type'] == 'LogicalAnd':
            tf_layers_dict[layer_id] = tf.math.logical_and(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])

        ### LogicalNot
        elif layer.attrib['type'] == 'LogicalNot':
            tf_layers_dict[layer_id] = tf.math.logical_not(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

        ### LogicalOr
        elif layer.attrib['type'] == 'LogicalOr':
            tf_layers_dict[layer_id] = tf.math.logical_or(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])

        ### LogicalXor
        elif layer.attrib['type'] == 'LogicalXor':
            tf_layers_dict[layer_id] = tf.math.logical_xor(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])

        ### Broadcast - TODO
        elif layer.attrib['type'] == 'Broadcast':
            mode = data.attrib['mode']
            if mode == 'numpy':
                tf_layers_dict[layer_id] = tf.broadcast_to(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])
            elif mode == 'bidirectional':
                tf_layers_dict[layer_id] = tf.math.multiply(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf.ones(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]))
            else:
                print('The {} mode of broadcast is not yet implemented.'.format(mode))
                sys.exit(-1)

        ### Split
        elif layer.attrib['type'] == 'Split':
            num_splits = int(data.attrib['num_splits'])
            axis = int(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])
            if axis == 1:
                axis = 3
            elif axis >= 2:
                axis -= 1

            outputs = tf.split(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], num_or_size_splits=num_splits, axis=axis)

            for output, layer_id_port in zip(outputs, layer_id_port_dict[layer_id]['layer_id:port']):
                tf_layers_dict[layer_id_port] = output

        ### VariadicSplit
        elif layer.attrib['type'] == 'VariadicSplit':
            axis = int(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])
            if axis == 1:
                axis = -1
            elif axis >= 2:
                axis -= 1
            num_or_size_splits = None
            if type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]) == np.ndarray:
                num_or_size_splits = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]
            else:
                # num_or_size_splits = [tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)].shape[idx] for idx, val in enumerate(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)])]
                num_or_size_splits = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]

            outputs = tf.split(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], num_or_size_splits=num_or_size_splits, axis=axis)

            for output, layer_id_port in zip(outputs, layer_id_port_dict[layer_id]['layer_id:port']):
                tf_layers_dict[layer_id_port] = output

        ### MVN
        elif layer.attrib['type'] == 'MVN':
            eps = float(data.attrib['eps'])
            across_channels = data.attrib['across_channels']
            # normalize_variance = data.attrib['normalize_variance']

            if across_channels == '0':
                across_channels = False
            elif across_channels == '1':
                across_channels = True
            elif across_channels == 'False':
                across_channels = False
            elif across_channels == 'True':
                across_channels = True

            # if normalize_variance == '0':
            #     normalize_variance = False
            # elif normalize_variance == '1':
            #     normalize_variance = True
            # elif normalize_variance == 'False':
            #     normalize_variance = False
            # elif normalize_variance == 'True':
            #     normalize_variance = True

            mean = None
            var = None
            x = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
            if across_channels:
                mean = tf.math.reduce_mean(x, axis=[-1], keepdims=True)
                var = tf.math.reduce_variance(x, axis=[-1], keepdims=True)
            else:
                mean = tf.math.reduce_mean(x, keepdims=True)
                var = tf.math.reduce_variance(x, keepdims=True)
            mvn = (x - mean) / tf.math.sqrt(var + eps)

            tf_layers_dict[layer_id] = mvn

        # ### NonMaxSuppression - TODO
        # elif layer.attrib['type'] == 'NonMaxSuppression':
        #     box_encoding = 'corner' # or center
        #     sort_result_descending = True
        #     output_type = 'i64'

        #     if not data is None and 'box_encoding' in data.attrib:
        #         box_encoding = data.attrib['box_encoding']
        #     if not data is None and 'sort_result_descending' in data.attrib:
        #         sort_result_descending = data.attrib['sort_result_descending']
        #     if not data is None and 'output_type' in data.attrib:
        #         output_type = data.attrib['output_type']

        #     boxes = None
        #     if box_encoding == 'center':
        #         boxes = []
        #         input_boxes = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] # [1, 1000, 4]
        #         for idx, x_center, y_center, width, height in enumerate(input_boxes[0]):
        #             x1 = x_center - (width // 2)
        #             y1 = y_center - (height // 2)
        #             x2 = x_center + (width // 2)
        #             y2 = y_center + (height // 2)
        #             boxes.append([x1, y1, x2, y2])
        #         boxes = np.asanyarray(boxes)[np.newaxis, :, :]
        #     else:
        #         boxes = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]


        ### Result
        elif layer.attrib['type'] == 'Result':
            tf_layers_dict[layer_id] = tf.identity(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], name=layer.attrib['name'].split('/')[0])
            tf_outputs.append(tf_layers_dict[layer_id])
        else:
            print('The {} layer is not yet implemented.'.format(layer.attrib['type']))
            sys.exit(-1)

        if debug and idx > debug_layer_number:
            tf_outputs.append(tf_layers_dict[layer_id])
            break


    model = Model(inputs=tf_inputs, outputs=tf_outputs)
    model.summary()


    # saved_model output
    if output_saved_model:
        try:
            tf.saved_model.save(model, model_output_path)
        except Exception as e:
            print(e)
            print('Switch to the output of an optimized protocol buffer file (.pb).')
            output_pb = True
            output_h5 = False

    # .h5 output
    if output_h5:
        model.save('{}/model_float32.h5'.format(model_output_path))

    # weight and json output
    if output_weight_and_json:
        open('{}/model_float32.json'.format(model_output_path), 'w').write(model.to_json())
        model.save_weights('{}/model_float32_weights.h5'.format(model_output_path))

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
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        with open('{}/model_float32.tflite'.format(model_output_path), 'wb') as w:
            w.write(tflite_model)
        print("tflite convert complete! - {}/model_float32.tflite".format(model_output_path))

    # Weight Quantization - Input/Output=float32
    if output_weight_quant_tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        with open('{}/model_weight_quant.tflite'.format(model_output_path), 'wb') as w:
            w.write(tflite_model)
        print('Weight Quantization complete! - {}/model_weight_quant.tflite'.format(model_output_path))

    # Float16 Quantization - Input/Output=float32
    if output_float16_quant_tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_quant_model = converter.convert()
        with open('{}/model_float16_quant.tflite'.format(model_output_path), 'wb') as w:
            w.write(tflite_quant_model)
        print('Float16 Quantization complete! - {}/model_float16_quant.tflite'.format(model_output_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='input IR model path (.xml)')
    parser.add_argument('--model_output_path', type=str, default='saved_model', help='The output folder path of the converted model file')
    parser.add_argument('--output_saved_model', type=bool, default=False, help='saved_model output switch')
    parser.add_argument('--output_h5', type=bool, default=False, help='.h5 output switch')
    parser.add_argument('--output_weight_and_json', type=bool, default=False, help='weight of h5 and json output switch')
    parser.add_argument('--output_pb', type=bool, default=False, help='.pb output switch')
    parser.add_argument('--output_no_quant_float32_tflite', type=bool, default=False, help='float32 tflite output switch')
    parser.add_argument('--output_weight_quant_tflite', type=bool, default=False, help='weight quant tflite output switch')
    parser.add_argument('--output_float16_quant_tflite', type=bool, default=False, help='float16 quant tflite output switch')
    parser.add_argument('--replace_swish_and_hardswish', type=bool, default=False, help='Replace swish and hard-swish with each other.')
    parser.add_argument('--replace_prelu_and_minmax', type=bool, default=False, help='Replace prelu and minimum/maximum with each other.')
    parser.add_argument('--debug', action='store_true', help='debug mode switch')
    parser.add_argument('--debug_layer_number', type=int, default=0, help='The last layer number to output when debugging. Used only when --debug=True.')
    args = parser.parse_args()
    model, ext = os.path.splitext(args.model_path)
    model_output_path = args.model_output_path.rstrip('/')
    if ext != '.xml':
        print('The specified model is not \'.xml\' file.')
        sys.exit(-1)
    output_saved_model = args.output_saved_model
    output_h5 = args.output_h5
    output_weight_and_json = args.output_weight_and_json
    output_pb = args.output_pb
    output_no_quant_float32_tflite =  args.output_no_quant_float32_tflite
    output_weight_quant_tflite = args.output_weight_quant_tflite
    output_float16_quant_tflite = args.output_float16_quant_tflite
    replace_swish_and_hardswish = args.replace_swish_and_hardswish
    replace_prelu_and_minmax = args.replace_prelu_and_minmax
    debug = args.debug
    debug_layer_number = args.debug_layer_number
    if not output_saved_model and \
        not output_h5 and \
        not output_weight_and_json and \
        not output_pb and \
        not output_no_quant_float32_tflite and \
        not output_weight_quant_tflite and \
        not output_float16_quant_tflite:
        print('Set at least one of the output switches (output_*) to true.')
        sys.exit(-1)
    os.makedirs(model_output_path, exist_ok=True)
    convert(model, model_output_path, output_saved_model, output_h5, output_weight_and_json, output_pb,
            output_no_quant_float32_tflite, output_weight_quant_tflite, output_float16_quant_tflite,
            replace_swish_and_hardswish, replace_prelu_and_minmax,
            debug, debug_layer_number)

if __name__ == "__main__":
    main()
