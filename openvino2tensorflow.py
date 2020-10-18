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
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, ReLU, PReLU, MaxPool2D, AveragePooling2D, Reshape, Concatenate, Conv2DTranspose, Layer
from tensorflow.keras.initializers import Constant
from tensorflow.keras.backend import shape, clip
from tensorflow.keras.activations import tanh, elu, sigmoid, swish, softmax
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
            output_float16_quant_tflite,
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

    transpose_squeeze_skip = False

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
                                              kernel_initializer=Constant(tf_layers_dict[tf_edges[layer_id][1]].transpose(2,3,1,0)))(tf_layers_dict[tf_edges[layer_id][0]])

        ### Add
        elif layer.attrib['type'] == 'Add':
            # 'Fused_Add_' == BiasAdd
            if len(tf_edges[layer_id]) == 2 and (type(tf_layers_dict[tf_edges[layer_id][1]]) == np.ndarray):
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

        ### PReLU
        elif layer.attrib['type'] == 'PReLU':
            tf_layers_dict[layer_id] = PReLU(alpha_initializer=Constant(tf_layers_dict[tf_edges[layer_id][1]]))(tf_layers_dict[tf_edges[layer_id][0]])

        ### Clamp
        elif layer.attrib['type'] == 'Clamp':
            cmin = float(data.attrib['min'])
            cmax = float(data.attrib['max'])
            if cmin == 0.0 and cmax == 6.0:
                # ReLU6
                tf_layers_dict[layer_id] = tf.nn.relu6(tf_layers_dict[tf_edges[layer_id][0]])
            else:
                # Other
                tf_layers_dict[layer_id] = clip(tf_layers_dict[tf_edges[layer_id][0]], min_value=cmin, max_value=cmax)

        ### Tan
        elif layer.attrib['type'] == 'Tan':
            tf_layers_dict[layer_id] = tf.math.tan(tf_layers_dict[tf_edges[layer_id][0]])

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

        ### Swish
        elif layer.attrib['type'] == 'Swish':
            tf_layers_dict[layer_id] = swish(tf_layers_dict[tf_edges[layer_id][0]])

        ### MaxPool
        elif layer.attrib['type'] == 'MaxPool':
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
            tf_layers_dict[layer_id] = tf.nn.max_pool(tf_layers_dict[tf_edges[layer_id][0]], ksize=kernel_size, strides=strides, padding=padding)

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
            tf_layers_dict[layer_id] = AveragePooling2D(pool_size=kernel_size, strides=strides, padding=padding)(tf_layers_dict[tf_edges[layer_id][0]])

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
                                                       kernel_initializer=Constant(tf_layers_dict[tf_edges[layer_id][1]].transpose(2,3,1,0)))(tf_layers_dict[tf_edges[layer_id][0]])

        ### Concat
        elif layer.attrib['type'] == 'Concat':
            tf_layers_dict[layer_id] = Concatenate()([tf_layers_dict[from_layer_id] for from_layer_id in tf_edges[layer_id]])

        ### Multiply
        elif layer.attrib['type'] == 'Multiply':
            if len(tf_edges[layer_id]) == 2 and (type(tf_layers_dict[tf_edges[layer_id][1]]) == np.ndarray):
                if tf_layers_dict[tf_edges[layer_id][1]].ndim == 4:
                    # 4D - NCHW->NHWC
                    tf_layers_dict[layer_id] = tf.math.multiply(tf_layers_dict[tf_edges[layer_id][0]], tf_layers_dict[tf_edges[layer_id][1]].transpose(0,2,3,1).astype(np.float32))
                else:
                    # unknown
                    tf_layers_dict[layer_id] = tf.math.multiply(tf_layers_dict[tf_edges[layer_id][0]], tf_layers_dict[tf_edges[layer_id][1]])

            elif len(tf_edges[layer_id]) == 2 and (type(tf_layers_dict[tf_edges[layer_id][0]]) == np.ndarray):
                if tf_layers_dict[tf_edges[layer_id][0]].ndim == 4:
                    # 4D - NCHW->NHWC
                    tf_layers_dict[layer_id] = tf.math.multiply(tf_layers_dict[tf_edges[layer_id][0]].transpose(0,2,3,1).astype(np.float32), tf_layers_dict[tf_edges[layer_id][1]])
                else:
                    # unknown
                    tf_layers_dict[layer_id] = tf.math.multiply(tf_layers_dict[tf_edges[layer_id][0]], tf_layers_dict[tf_edges[layer_id][1]])

            else:
                # unknown
                tf_layers_dict[layer_id] = tf.math.multiply(tf_layers_dict[tf_edges[layer_id][0]], tf_layers_dict[tf_edges[layer_id][1]])

        ### Interpolate
        elif layer.attrib['type'] == 'Interpolate':
            mode = data.attrib['mode']
            antialias = False if int(data.attrib['antialias']) == 0 else True
            out_port0 = [int(sdim.text) for sdim in layer.find('output')[0]]
            out_height = int(out_port0[2])
            out_width  = int(out_port0[3])
            if mode == 'linear':
                tf_layers_dict[layer_id] = tf.image.resize(tf_layers_dict[tf_edges[layer_id][0]], [out_height, out_width], method='bilinear', antialias=antialias)
            elif mode == 'nearest':
                tf_layers_dict[layer_id] = tf.image.resize(tf_layers_dict[tf_edges[layer_id][0]], [out_height, out_width], method='nearest', antialias=antialias)
            else:
                print('The Interpolate - {} is not yet implemented.'.format(mode))
                sys.exit(-1)


        ### ShapeOf
        elif layer.attrib['type'] == 'ShapeOf':
            tf_layers_dict[layer_id] = tf.keras.backend.shape(tf_layers_dict[tf_edges[layer_id][0]])

        ### Convert
        elif layer.attrib['type'] == 'Convert':
            # vino:    u8,    u16,    u32,    u64,   i8,   i16,   i32,   i64,     f16,     f32,              bf16, boolean
            # tf  : uint8, uint16, uint32, uint64, int8, int16, int32, int64, float16, float32, float64, bfloat16
            destination_type = data.attrib['destination_type']
            tf_layers_dict[layer_id] = tf.cast(tf_layers_dict[tf_edges[layer_id][0]], cast_type_ov_tf[destination_type])

        ### StridedSlice
        elif layer.attrib['type'] == 'StridedSlice':
            begin_mask       = int(data.attrib['begin_mask'])
            end_mask         = int(data.attrib['end_mask'])
            ellipsis_mask    = int(data.attrib['ellipsis_mask'])
            new_axis_mask    = int(data.attrib['new_axis_mask'])
            shrink_axis_mask = int(data.attrib['shrink_axis_mask'])

            begin   = [-1] if int(tf_layers_dict[tf_edges[layer_id][1]]) == -1 else [int(tf_layers_dict[tf_edges[layer_id][1]]) - 1]
            end     = [-1] if int(tf_layers_dict[tf_edges[layer_id][2]]) == -1 else [int(tf_layers_dict[tf_edges[layer_id][2]]) - 1]
            strides = [int(tf_layers_dict[tf_edges[layer_id][3]])]
            tf_layers_dict[layer_id] = tf.strided_slice(tf_layers_dict[tf_edges[layer_id][0]],
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
            pads_begin = tf_layers_dict[tf_edges[layer_id][1]] # [0,0,1,1]
            pads_end   = tf_layers_dict[tf_edges[layer_id][2]] # [0,0,1,1]

            pad_b_top    = 0 if (pads_end[0] == 0 and pads_begin[0] == 0) else  (pads_end[0] - pads_begin[0] + 1)
            pad_b_bottom = pad_b_top

            pad_c_top    = 0 if (pads_end[1] == 0 and pads_begin[1] == 0) else  (pads_end[1] - pads_begin[1] + 1)
            pad_c_bottom = pad_c_top

            pad_top    = 0 if (pads_end[2] == 0 and pads_begin[2] == 0) else  (pads_end[2] - pads_begin[2] + 1)
            pad_bottom = pad_top
            pad_left   = 0 if (pads_end[3] == 0 and pads_begin[3] == 0) else  (pads_end[3] - pads_begin[3] + 1)
            pad_right  = pad_left
            paddings = [[pad_b_top, pad_b_bottom], [pad_top, pad_bottom], [pad_left, pad_right], [pad_c_top, pad_c_bottom]]
            pad_value  = [0.0]
            if 'pad_value' in data.attrib:
                pad_value = [float(data.attrib['pad_value'])]
            tf_layers_dict[layer_id] = tf.pad(tf_layers_dict[tf_edges[layer_id][0]], paddings, mode=pad_mode, constant_values=pad_value)

        ### TopK
        elif layer.attrib['type'] == 'TopK':
            # axis = int(data.attrib['axis'])
            # index_element_type = data.attrib['index_element_type']
            # mode = data.attrib['mode']
            # sort = data.attrib['sort']
            transpose_squeeze_skip = False
            k = int(tf_layers_dict[tf_edges[layer_id][1]])
            if k == 1:
                tf_layers_dict[layer_id] = tf.math.argmax(tf_layers_dict[tf_edges[layer_id][0]], axis=-1, output_type=tf.dtypes.int32)
                transpose_squeeze_skip = True
                temp_final_output_layer = tf_layers_dict[layer_id]
            else:
                tf_layers_dict[layer_id] = tf.math.top_k(tf_layers_dict[tf_edges[layer_id][0]], k=int(tf_layers_dict[tf_edges[layer_id][1]]), sorted=True)
                transpose_squeeze_skip = False

        ### Transpose
        elif layer.attrib['type'] == 'Transpose':
            if transpose_squeeze_skip and tf_layers_dict[tf_edges[layer_id][0]].name.split(':')[0] == 'ArgMax':
                continue
            else:
                tf_layers_dict[layer_id] = tf.transpose(tf_layers_dict[tf_edges[layer_id][0]], perm=tf_layers_dict[tf_edges[layer_id][1]])

        ### Squeeze
        elif layer.attrib['type'] == 'Squeeze':
            if transpose_squeeze_skip:
                continue
            else:
                tf_layers_dict[layer_id] = tf.squeeze(tf_layers_dict[tf_edges[layer_id][0]], axis=int(tf_layers_dict[tf_edges[layer_id][1]]))
            transpose_squeeze_skip = False

        ### Gather
        elif layer.attrib['type'] == 'Gather':
            tf_layers_dict[layer_id] = tf.gather(tf_layers_dict[tf_edges[layer_id][0]], int(tf_layers_dict[tf_edges[layer_id][1]]), axis=tf_layers_dict[tf_edges[layer_id][2]])

        ### ReduceMean, ReduceMax, ReduceMin, ReduceSum
        elif layer.attrib['type'] == 'ReduceMean' or layer.attrib['type'] == 'ReduceMax' or layer.attrib['type'] == 'ReduceMin' or layer.attrib['type'] == 'ReduceSum':
            keep_dims = True if data.attrib['keep_dims'] == "True" else False
            if (type(tf_layers_dict[tf_edges[layer_id][1]]) == np.ndarray) and len(tf_layers_dict[tf_edges[layer_id][1]]) == 2 and \
               ((tf_layers_dict[tf_edges[layer_id][1]][0] == 3 and tf_layers_dict[tf_edges[layer_id][1]][1] == 2) or (tf_layers_dict[tf_edges[layer_id][1]][0] == 2 and tf_layers_dict[tf_edges[layer_id][1]][1] == 3)):
                axis1 = tf_layers_dict[tf_edges[layer_id][1]][0] - 1
                axis2 = tf_layers_dict[tf_edges[layer_id][1]][1] - 1
            else:
                axis1 = tf_layers_dict[tf_edges[layer_id][1]][0]
                axis2 = tf_layers_dict[tf_edges[layer_id][1]][1]
            if layer.attrib['type'] == 'ReduceMean':
                tf_layers_dict[layer_id] = tf.math.reduce_mean(tf_layers_dict[tf_edges[layer_id][0]], axis=[axis1, axis2], keepdims=keep_dims)
            elif layer.attrib['type'] == 'ReduceMax':
                tf_layers_dict[layer_id] = tf.math.reduce_max(tf_layers_dict[tf_edges[layer_id][0]], axis=[axis1, axis2], keepdims=keep_dims)
            elif layer.attrib['type'] == 'ReduceMin':
                tf_layers_dict[layer_id] = tf.math.reduce_min(tf_layers_dict[tf_edges[layer_id][0]], axis=[axis1, axis2], keepdims=keep_dims)
            elif layer.attrib['type'] == 'ReduceSum':
                tf_layers_dict[layer_id] = tf.math.reduce_sum(tf_layers_dict[tf_edges[layer_id][0]], axis=[axis1, axis2], keepdims=keep_dims)

        ### MatMul
        elif layer.attrib['type'] == 'MatMul':
            if not data is None and 'a' in data.attrib:
                transpose_a = True if int(data.attrib['a']) == 1 else False
            if not data is None and 'b' in data.attrib:
                transpose_b = True if int(data.attrib['b']) == 1 else False
            if not data is None and 'transpose_a' in data.attrib:
                transpose_a = True if int(data.attrib['transpose_a']) == 1 else False
            if not data is None and 'transpose_b' in data.attrib:
                transpose_b = True if int(data.attrib['transpose_b']) == 1 else False
            tf_layers_dict[layer_id] = tf.linalg.matmul(tf_layers_dict[tf_edges[layer_id][0]], tf_layers_dict[tf_edges[layer_id][1]],
                                                        transpose_a, transpose_b)

        ### Reshape
        elif layer.attrib['type'] == 'Reshape':
            tf_layers_dict[layer_id] = tf.reshape(tf_layers_dict[tf_edges[layer_id][0]], tf_layers_dict[tf_edges[layer_id][1]])

        ### Range
        elif layer.attrib['type'] == 'Range':
            dtype = cast_type_ov_tf[data.attrib['output_type']]
            tf_layers_dict[layer_id] = tf.range(tf_layers_dict[tf_edges[layer_id][0]][0], tf_layers_dict[tf_edges[layer_id][1]],
                                                delta=int(tf_layers_dict[tf_edges[layer_id][2]]), dtype=dtype)

        ### Exp
        elif layer.attrib['type'] == 'Exp':
            tf_layers_dict[layer_id] = tf.math.exp(tf_layers_dict[tf_edges[layer_id][0]])

        ### Abs
        elif layer.attrib['type'] == 'Abs':
            tf_layers_dict[layer_id] = tf.math.abs(tf_layers_dict[tf_edges[layer_id][0]])

        ### SoftMax
        elif layer.attrib['type'] == 'SoftMax':
            axis = int(data.attrib['axis'])
            tf_layers_dict[layer_id] = softmax(tf_layers_dict[tf_edges[layer_id][0]], axis=axis)

        ### Negative
        elif layer.attrib['type'] == 'Negative':
            tf_layers_dict[layer_id] = tf.math.negative(tf_layers_dict[tf_edges[layer_id][0]])

        ### Maximum
        elif layer.attrib['type'] == 'Maximum':
            # No broadcast
            tf_layers_dict[layer_id] = tf.math.maximum(tf_layers_dict[tf_edges[layer_id][0]], tf_layers_dict[tf_edges[layer_id][1]])

        ### Minimum
        elif layer.attrib['type'] == 'Minimum':
            # No broadcast
            tf_layers_dict[layer_id] = tf.math.minimum(tf_layers_dict[tf_edges[layer_id][0]], tf_layers_dict[tf_edges[layer_id][1]])

        ### Acos
        elif layer.attrib['type'] == 'Acos':
            tf_layers_dict[layer_id] = tf.math.acos(tf_layers_dict[tf_edges[layer_id][0]])

        ### Acosh
        elif layer.attrib['type'] == 'Acosh':
            tf_layers_dict[layer_id] = tf.math.acosh(tf_layers_dict[tf_edges[layer_id][0]])

        ### Asin
        elif layer.attrib['type'] == 'Asin':
            tf_layers_dict[layer_id] = tf.math.asin(tf_layers_dict[tf_edges[layer_id][0]])

        ### Asinh
        elif layer.attrib['type'] == 'Asinh':
            tf_layers_dict[layer_id] = tf.math.asinh(tf_layers_dict[tf_edges[layer_id][0]])

        ### Atan
        elif layer.attrib['type'] == 'Atan':
            tf_layers_dict[layer_id] = tf.math.atan(tf_layers_dict[tf_edges[layer_id][0]])

        ### Atanh
        elif layer.attrib['type'] == 'Atanh':
            tf_layers_dict[layer_id] = tf.math.atanh(tf_layers_dict[tf_edges[layer_id][0]])

        ### Result
        elif layer.attrib['type'] == 'Result':
            try:
                tf_layers_dict[layer_id] = tf.identity(tf_layers_dict[tf_edges[layer_id][0]], name=layer.attrib['name'].split('/')[0])
                tf_outputs.append(tf_layers_dict[layer_id])
            except:
                tf_layers_dict[layer_id] = tf.identity(temp_final_output_layer, name=layer.attrib['name'].split('/')[0])
                tf_outputs.append(tf_layers_dict[layer_id])
                transpose_squeeze_skip = False
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
    output_pb = args.output_pb
    output_no_quant_float32_tflite =  args.output_no_quant_float32_tflite
    output_weight_quant_tflite = args.output_weight_quant_tflite
    output_float16_quant_tflite = args.output_float16_quant_tflite
    debug = args.debug
    debug_layer_number = args.debug_layer_number
    if not output_saved_model and \
        not output_h5 and \
        not output_pb and \
        not output_no_quant_float32_tflite and \
        not output_weight_quant_tflite and \
        not output_float16_quant_tflite:
        print('Set at least one of the output switches (output_*) to true.')
        sys.exit(-1) 
    convert(model, model_output_path, output_saved_model, output_h5, output_pb,
            output_no_quant_float32_tflite, output_weight_quant_tflite, output_float16_quant_tflite,
            debug, debug_layer_number)

if __name__ == "__main__":
    main()