#! /usr/bin/env python
'''
tensorflow==2.8.0+

python3 openvino2tensorflow.py \
--model_path openvino/448x448/FP32/Resnet34_3inputs_448x448_20200609.xml \
--output_saved_model \
--output_pb \
--output_weight_quant_tflite \
--output_float16_quant_tflite \
--output_no_quant_float32_tflite \
--non_verbose

python3 openvino2tensorflow.py \
--model_path debug/openvino/yolox_nano/320x320/FP32/yolox_nano_320x320.xml \
--output_saved_model \
--output_pb \
--output_no_quant_float32_tflite \
--weight_replacement_config debug/weight_replacement_config_yolox_nano.json \
--non_verbose
'''
import os
import sys
import argparse
import struct
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as et
import logging
import warnings

from tensorflow.python.framework.ops import _run_using_default_session
from tensorflow.python.keras.backend import ndim
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

def convert(
    model_path,
    model_output_path,
    output_saved_model,
    output_h5,
    output_weight_and_json,
    output_pb,
    output_no_quant_float32_tflite,
    output_dynamic_range_quant_tflite,
    output_weight_quant_tflite,
    output_float16_quant_tflite,
    output_integer_quant_tflite,
    output_full_integer_quant_tflite,
    output_integer_quant_type,
    string_formulas_for_normalization,
    calib_ds_type,
    ds_name_for_tfds_for_calibration,
    split_name_for_tfds_for_calibration,
    download_dest_folder_path_for_the_calib_tfds,
    tfds_download_flg,
    npy_load_default_path,
    load_dest_file_path_for_the_calib_npy,
    output_tfjs,
    output_tftrt_float32,
    output_tftrt_float16,
    tftrt_maximum_cached_engines,
    output_coreml,
    output_edgetpu,
    edgetpu_compiler_timeout,
    edgetpu_num_segments,
    output_onnx,
    onnx_opset,
    onnx_extra_opset,
    use_onnx_nchw_conversion,
    use_onnx_optimization,
    output_myriad,
    vpu_number_of_shaves,
    vpu_number_of_cmx_slices,
    replace_swish_and_hardswish,
    optimizing_hardswish_for_edgetpu,
    replace_prelu_and_minmax,
    replace_argmax,
    replace_argmax_indices_to_float32,
    restricted_resize_image_mode,
    weight_replacement_config,
    use_experimental_new_quantizer,
    optimizing_barracuda,
    layerids_of_the_terminating_output,
    keep_input_tensor_in_nchw,
    verbose
):

    print(f'{Color.REVERCE}TensorFlow/Keras model building process starts{Color.RESET}', '=' * 38)

    import subprocess
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    tf.get_logger().setLevel(logging.ERROR)
    import tensorflow_datasets as tfds
    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, AveragePooling2D, Conv2DTranspose, PReLU, Lambda, LeakyReLU, Conv3D
    from tensorflow.keras.initializers import Constant
    from tensorflow.keras.activations import elu
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    from tensorflow.python.framework.ops import EagerTensor
    if output_coreml:
        import coremltools as ct
    import json
    import pprint
    import math

    # for unpacking binary buffer
    format_config = {
        'FP32' : ['f', 4],
        'FP16' : ['e', 2],
        'I64'  : ['q', 8],
        'I32'  : ['i', 4],
        'I16'  : ['h', 2],
        'I8'   : ['b', 1],
        'U8'   : ['B', 1],
        'BOOL' : ['?', 1]
    }

    # vino:    u8,    u16,    u32,    u64,   i8,   i16,   i32,   i64,     f16,     f32,              bf16, boolean
    # tf  : uint8, uint16, uint32, uint64, int8, int16, int32, int64, float16, float32, float64, bfloat16

    # type conversion table
    cast_type_ov_tf = {
        'u8'  : tf.uint8,
        'u16' : tf.uint16,
        'u32' : tf.uint32,
        'u64' : tf.uint64,
        'i8'  : tf.int8,
        'i16' : tf.int16,
        'i32' : tf.int32,
        'i64' : tf.int64,
        'f16' : tf.float16,
        'f32' : tf.float32,
        'bf16': tf.bfloat16,
        'boolean': tf.bool
    }

    # integer type table
    int_type_tf = [
        tf.uint8,
        tf.uint16,
        tf.uint32,
        tf.uint64,
        tf.int8,
        tf.int16,
        tf.int32,
        tf.int64
    ]

    # numpy type convertion table
    cast_type_np_tf = {
        'uint8'   : tf.uint8,
        'uint16'  : tf.uint16,
        'uint32'  : tf.uint32,
        'uint64'  : tf.uint64,
        'int8'    : tf.int8,
        'int16'   : tf.int16,
        'int32'   : tf.int32,
        'int64'   : tf.int64,
        'float16' : tf.float16,
        'float32' : tf.float32,
        'float64' : tf.float64,
        'bool'    : tf.bool
    }

    # pad type conversion table
    pad_type_ov_tf = {
        'constant' : 'CONSTANT',
        'reflect'  : 'REFLECT',
        'symmetric': 'SYMMETRIC',
        'edge'     : 'REFLECT'
    }

    # op_type to ignore garbage output with opset=8
    ignore_garbage_output_op_types = [
        'MaxPool',
    ]

    # Read IR weight data
    with open(model_path+'.bin', 'rb') as f:
        binWeight = f.read()
    # Parse IR XML file,
    tree = et.parse(model_path+'.xml')
    root = tree.getroot()
    edges = root.find('edges')
    layers = root.find('layers')
    tf_layers_dict = {}
    tf_edges = {}

    tf_inputs = []
    tf_outputs = []
    layer_id_port_dict = {}

    def is_integer_num(n):
        if isinstance(n, int):
            return True
        if isinstance(n, float):
            return n.is_integer()
        return False

    def get_num_of_outputs_per_layer_id(tf_edges):
        output_count_by_layer_id_tmp = {}
        for key in tf_edges.keys():
            key_tmp = key.split(':')[0]
            output_count_by_layer_id_tmp.setdefault(key_tmp, {'count': 0, 'layer_id:port': []})
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


    """
    format_version : Format version of weight_replacement_config.
    layer_id : ID of the Const layer whose weight/constant parameter is to be swapped.
                For example, specify "1123" for layer id="1123" for type="Const" in .xml.

                <layer id="1123" name="Decoder/softmax/Reshape_1/Cast_123722_const657_const" type="Const" version="opset1">
                    <data element_type="i64" offset="7632604" shape="4" size="32"/>
                    <output>
                        <port id="1" precision="I64">
                            <dim>4</dim>
                        </port>
                    </output>
                </layer>

    replace_mode : "direct" or "npy"
                    "direct": Specify the values of the Numpy matrix directly in the "values" attribute.
                            Ignores the values recorded in the .bin file and replaces them with the values specified in "values".

                    {
                        "layer_id": "1123",
                        "replace_mode": "direct",
                        "values": [
                            1,
                            2,
                            513,
                            513
                        ]
                    }

                    "npy": Load a Numpy binary file with the matrix output by np.save('xyz', a).
                        The "values" attribute specifies the path to the Numpy binary file.
                    {
                        "layer_id": "1125",
                        "replace_mode": "npy",
                        "values": "weights/xyz.npy"
                    }

    values : Specify the value or the path to the Numpy binary file to replace the weight/constant value recorded in .bin.
            The way to specify is as described in the description of 'replace_mode'.
    """
    # Elements for each version of weights_replacement_config
    # key = config version
    # value = Allowed elements for each version
    weights_replacement_config_version_elements = {
        1 : ['layer_id', 'replace_mode', 'values'],
        2 : ['layer_id', 'type', 'replace_mode', 'values']
    }
    # Combinations of possible values for type key and replace_mode in weights_replacement_config.
    # key = Type name
    # value = List of replace_mode
    weights_replacement_config_types = {
        'Const': ['direct', 'npy'],
        'Transpose': ['insert_before', 'insert_after'],
        'Reshape': ['insert_before', 'insert_after'],
        'Cast': ['insert_before', 'insert_after'],
        'Concat': ['change_axis'],
        'SoftMax': ['change_axis'],
        'ShuffleChannels': ['change_axis'],
        'StridedSlice': ['change_attributes', 'replace'],
        'MaxPool': ['change_padding_mode'],
        'PReLU': ['change_shared_axes'],
        'ReverseSequence': ['change_batch_axis', 'change_seq_axis'],
        'Squeeze': ['insert_before', 'insert_after'],
        'Unsqueeze': ['insert_before', 'insert_after'],
        'Einsum': ['change_equation'],
        'Add': ['insert_before', 'insert_after'],
        'Multiply': ['insert_before', 'insert_after'],
    }


    def parse_json(jsonfile_path: str):
        """Parsing weights_replacement_config

        Args:
        ----------
            jsonfile_path : str
                Path to the weights_replacement_config file

        Returns:
        ----------
            format_version : int
                Format version number of weights_replacement_config

            layers : dict
                Result of parsing weights_replacement_config into dict format
        """
        j = json.load(open(jsonfile_path))
        format_version = j['format_version']
        layers = {}
        for v in j['layers']:
            # Elements check
            for k in v.keys():
                if not k in weights_replacement_config_version_elements[format_version]:
                    key_name1 = 'layer_id'
                    print(f'{Color.RED}ERROR:{Color.RESET} It contains a key that cannot be included in the config with format_version: {format_version}. layer_id: {v[key_name1]}, key: "{k}"')
                    print(f'{Color.RED}ERROR:{Color.RESET} List of keys to allow in format_version: {format_version}. {weights_replacement_config_version_elements[format_version]}')
                    sys.exit(-1)
            for k in weights_replacement_config_version_elements[format_version]:
                if not k in v.keys():
                    key_name1 = 'layer_id'
                    print(f'{Color.RED}ERROR:{Color.RESET} Missing elements that must be included in the config for format_version: {format_version}. layer_id: {v[key_name1]}, key: "{k}"')
                    print(f'{Color.RED}ERROR:{Color.RESET} List of elements that must be included in the config for format_version: {format_version}. {weights_replacement_config_version_elements[format_version]}')
                    sys.exit(-1)
            # weights_replacement_config_types check (Only when format_version is 2 or higher)
            if format_version >= 2:
                # Type check
                if not v['type'] in weights_replacement_config_types.keys():
                    key_name1 = 'layer_id'
                    key_name2 = 'type'
                    print(f'{Color.RED}ERROR:{Color.RESET} It contains a key that cannot be included in the config. layer_id: {v[key_name1]}, type: "{v[key_name2]}"')
                    print(f'{Color.RED}ERROR:{Color.RESET} List of keys to allow. {weights_replacement_config_types.keys()}')
                    sys.exit(-1)
                # Replace Mode check
                if not v['replace_mode'] in weights_replacement_config_types[v['type']]:
                    key_name1 = 'layer_id'
                    key_name2 = 'replace_mode'
                    key_name3 = 'type'
                    print(f'{Color.RED}ERROR:{Color.RESET} It contains a key that cannot be included in the config. layer_id: {v[key_name1]}, replace_mode: "{v[key_name2]}"')
                    print(f'{Color.RED}ERROR:{Color.RESET} List of keys to allow. {weights_replacement_config_types[v[key_name3]]}')
                    sys.exit(-1)

            layers[v['layer_id']] = v

        print(f'{Color.GREEN}weight_replacement_config format_version:{Color.RESET} {format_version}')
        print(f'{Color.GREEN}Replace the value of Const for each layer_id with the value below.{Color.RESET}')
        pprint.pprint(layers)
        return format_version, layers

    format_version = None
    wr_config = None
    if weight_replacement_config:
        format_version, wr_config = parse_json(weight_replacement_config)


    def extrapolation_of_layers(setting_up_layers_to_be_extrapolated: dict, input):
        """Processing of input operations based on weights_replacement_config settings

        Args:
        ----------
            setting_up_layers_to_be_extrapolated : dict
                wr_config[layer_id]

                {
                    "layer_id": "659",
                    "type": "Transpose",
                    "replace_mode": "insert_before",
                    "values": [0,2,1]
                }

            input : INPUT operation
                INPUT layer to be input to TF operations

        Returns:
        ----------
            Processed input operations
        """
        tf_layer = None
        layer_type = setting_up_layers_to_be_extrapolated['type']
        param = setting_up_layers_to_be_extrapolated['values']

        if layer_type == 'Transpose':
            tf_layer = tf.transpose(
                input,
                perm=param
            )
        elif layer_type == 'Reshape':
            tf_layer = tf.reshape(
                input,
                shape=param
            )
        elif layer_type == 'Cast':
            tf_layer = tf.cast(
                input,
                dtype=cast_type_ov_tf[param]
            )
        elif layer_type == 'Squeeze':
            tf_layer = tf.squeeze(
                input,
                axis=param
            )
        elif layer_type == 'Unsqueeze':
            tf_layer = tf.expand_dims(
                input,
                axis=param
            )
        elif layer_type == 'Add':
            tf_layer = tf.math.add(
                input,
                param
            )
        elif layer_type == 'Multiply':
            tf_layer = tf.math.multiply(
                input,
                param
            )
        return tf_layer


    print(f'{Color.REVERCE}Layer structure{Color.RESET}', '=' * 69)
    def layer_structure_print(info: dict) -> None:
        for key, value in info.items():
            print(f'{Color.GREEN}{key}{Color.RESET}: {value}')
        print('=' * 84)

    # edges
    added_key_list = []
    concat_port_list = {}
    for edge in edges:
        to_layer = edge.attrib['to-layer']
        to_layer_port = edge.attrib['to-port']
        from_layer = edge.attrib['from-layer']
        from_layer_port = edge.attrib['from-port']

        for layer in layers:
            if layer.attrib['id'] == to_layer:
                output_layer_ports = layer.find('output')

                if layer.attrib['type'] != 'Result' and len(output_layer_ports) >= 2 and layer.attrib['type'] not in ignore_garbage_output_op_types:
                    for port in output_layer_ports:
                        tf_edges.setdefault('{}:{}'.format(to_layer, port.attrib['id']), []).append(from_layer)
                    added_key_list.append(to_layer)
                else:
                    tf_edges.setdefault(to_layer, [])
                    if layer.attrib['type'] == 'Concat' or \
                        layer.attrib['type'] == 'Gather' or \
                            layer.attrib['type'] == 'GatherND' or \
                                layer.attrib['type'] == 'GatherElements' or \
                                    layer.attrib['type'] == 'ScatterElementsUpdate' or \
                                        layer.attrib['type'] == 'ScatterNDUpdate' or \
                                            layer.attrib['type'] == 'Reshape' or \
                                                layer.attrib['type'] == 'ConvertLike' or \
                                                    layer.attrib['type'] == 'Subtract' or \
                                                        layer.attrib['type'] == 'Divide' or \
                                                            layer.attrib['type'] == 'FloorMod' or \
                                                                layer.attrib['type'] == 'Power' or \
                                                                    layer.attrib['type'] == 'MatMul' or \
                                                                        layer.attrib['type'] == 'Greater' or \
                                                                            layer.attrib['type'] == 'GreaterEqual' or \
                                                                                layer.attrib['type'] == 'Less' or \
                                                                                    layer.attrib['type'] == 'LessEqual' or \
                                                                                        layer.attrib['type'] == 'SquaredDifference' or \
                                                                                            layer.attrib['type'] == 'PriorBox' or \
                                                                                                layer.attrib['type'] == 'PriorBoxClustered' or \
                                                                                                    layer.attrib['type'] == 'StridedSlice' or \
                                                                                                        layer.attrib['type'] == 'Select' or \
                                                                                                            layer.attrib['type'] == 'VariadicSplit' or \
                                                                                                                layer.attrib['type'] == 'ReverseSequence' or \
                                                                                                                    layer.attrib['type'] == 'Range' or \
                                                                                                                        layer.attrib['type'] == 'Einsum' or \
                                                                                                                            layer.attrib['type'] == 'ScatterUpdate':
                        concat_port_list.setdefault(to_layer, []).append(f'{from_layer}:{to_layer_port}')

        for layer in layers:
            if layer.attrib['id'] == from_layer:
                output_layer_ports = layer.find('output')
                if len(output_layer_ports) >= 2 and layer.attrib['type'] not in ignore_garbage_output_op_types:
                    flg = 'not_found'
                    for key in tf_edges.keys():
                        if to_layer in key and from_layer in tf_edges[key] and '{}:{}'.format(from_layer, from_layer_port) not in tf_edges[key]:
                            tf_edges[key].append('{}:{}'.format(from_layer, from_layer_port))
                            flg = 'found'
                            try:
                                tf_edges[key].remove(from_layer)
                            except:
                                pass

                    if flg == 'not_found':
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

    # The following loop sorts tf_edges in ascending order by port
    for to_layer, from_layer_ports in concat_port_list.items():
        temp_sorted_tf_edge = []
        # from_layer_ports = [from_layer_id:port, from_layer_id:port, from_layer_id:port, ...]
        ports = [p.split(':')[1] for p in from_layer_ports]
        for idx, port in enumerate(ports):
            temp_sorted_tf_edge.append(tf_edges[to_layer][ports.index(str(idx))])
        tf_edges[to_layer] = temp_sorted_tf_edge

    del added_key_list
    del concat_port_list

    layer_id_port_dict = get_num_of_outputs_per_layer_id(tf_edges)

    # layers
    for idx, layer in enumerate(layers):

        layer_id = layer.attrib['id']
        layer_name = layer.attrib['name'].replace('.', '_').replace('/', '_')
        data = layer.find('data')

        try:
            outputs = None
            layer_id_values = None
            layer_id_indices = None

            ### Parameter
            if layer.attrib['type'] == 'Parameter':
                if not data is None and 'shape' in data.attrib:
                    shape_str  = data.attrib['shape'].split(',')
                    shape = [int(s) for s in shape_str]
                    if len(shape) == 4:
                        if not keep_input_tensor_in_nchw:
                            tf_layers_dict[layer_id] = Input(shape=(shape[2], shape[3], shape[1]), batch_size=shape[0], name=layer_name)
                        else:
                            nchw = Input(shape=(shape[1], shape[2], shape[3]), batch_size=shape[0], name=layer_name)
                            tf_layers_dict[layer_id] = tf.transpose(nchw, perm=[0,2,3,1])
                    else:
                        if keep_input_tensor_in_nchw:
                            print(f'{Color.RED}ERROR:{Color.RESET} The keep_input_tensor_in_nchw parameter only supports 4D input. layer_id: {layer_id} input_shape: {shape}')
                            sys.exit(-1)
                        tf_layers_dict[layer_id] = Input(shape=[inp for inp in shape[1:]], batch_size=shape[0], name=layer_name)

                    if keep_input_tensor_in_nchw:
                        tf_inputs.append(nchw)
                    else:
                        tf_inputs.append(tf_layers_dict[layer_id])

                    layer_structure_print(
                        {
                            'layer_type': 'Input',
                            'layer_id': layer_id,
                            'tf_layers_dict': tf_layers_dict[layer_id]
                        }
                    )

                    if wr_config and layer_id in wr_config and format_version >= 2:
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to "Parameter" is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

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

                        if not wr_config or layer_id not in wr_config:
                            if type(decodedwgt) == np.ndarray and decodedwgt.dtype == np.float64:
                                tf_layers_dict[layer_id] = decodedwgt.astype(np.float32)
                            else:
                                tf_layers_dict[layer_id] = decodedwgt
                        else:
                            if layer_id in wr_config and format_version == 1:
                                if wr_config[layer_id]['replace_mode'] == 'direct':
                                    try:
                                        tf_layers_dict[layer_id] = np.array(wr_config[layer_id]['values'])
                                    except:
                                        tf_layers_dict[layer_id] = wr_config[layer_id]['values']
                                elif wr_config[layer_id]['replace_mode'] == 'npy':
                                    tf_layers_dict[layer_id] = np.load(wr_config[layer_id]['values'])

                            elif layer_id in wr_config and format_version >= 2 and wr_config[layer_id]['type'] == 'Const':
                                if wr_config[layer_id]['replace_mode'] == 'direct':
                                    try:
                                        tf_layers_dict[layer_id] = np.array(wr_config[layer_id]['values'])
                                    except:
                                        tf_layers_dict[layer_id] = wr_config[layer_id]['values']
                                elif wr_config[layer_id]['replace_mode'] == 'npy':
                                    tf_layers_dict[layer_id] = np.load(wr_config[layer_id]['values'])

                            else:
                                if type(decodedwgt) == np.ndarray and decodedwgt.dtype == np.float64:
                                    tf_layers_dict[layer_id] = decodedwgt.astype(np.float32)
                                else:
                                    tf_layers_dict[layer_id] = decodedwgt

                        if wr_config and layer_id in wr_config and format_version >= 2:
                            if wr_config[layer_id]['replace_mode'] == 'insert_before':
                                print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                                sys.exit(-1)

                            elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                                tf_layers_dict[layer_id] = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    tf_layers_dict[layer_id]
                                )
                        else:
                            pass

                        if verbose:
                            layer_structure_print(
                                {
                                    'layer_type': layer.attrib['type'],
                                    'layer_id': layer_id,
                                    'tf_layers_dict_shape': tf_layers_dict[layer_id].shape,
                                    'tf_layers_dict_value': tf_layers_dict[layer_id],
                                }
                            )
                        else:
                            layer_structure_print(
                                {
                                    'layer_type': layer.attrib['type'],
                                    'layer_id': layer_id,
                                    'tf_layers_dict_shape': tf_layers_dict[layer_id].shape,
                                }
                            )

            ### Convolution
            elif layer.attrib['type'] == 'Convolution':
                # port0 = [int(sdim.text) for sdim in layer.find('input')[0]]
                port1 = [int(sdim.text) for sdim in layer.find('input')[1]]
                filters = int(port1[0])
                dilations = [int(s) for s in data.attrib['dilations'].split(',')]
                strides = [int(s) for s in data.attrib['strides'].split(',')]
                pads_begin = 0
                pads_end = 0
                if not data is None and 'pads_begin' in data.attrib:
                    pads_begin = sum([int(s) for s in data.attrib['pads_begin'].split(',')])
                if not data is None and 'pads_end' in data.attrib:
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

                temp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]

                if len(strides) == 2:
                    # Conv2D
                    kernel_size = [int(port1[2]), int(port1[3])]
                    orig = None
                    if pads_begin > 0:
                        padding = 'valid'
                        # begin 0 = top
                        # begin 1 = left
                        # end 0 = bottom
                        # end 1 = right
                        begin = [int(data.attrib['pads_begin'].split(',')[0]), int(data.attrib['pads_end'].split(',')[0])]
                        end   = [int(data.attrib['pads_begin'].split(',')[1]), int(data.attrib['pads_end'].split(',')[1])]
                        orig = tf.keras.layers.ZeroPadding2D([begin, end])(temp)
                    else:
                        if temp.shape[0] == 1 and temp.shape[2] == 1 and temp.shape[3] == 1:
                            orig = tf.transpose(temp, perm=(0,2,3,1))
                        else:
                            orig = temp

                    try:
                        if wr_config and layer_id in wr_config and format_version >= 2:
                            if wr_config[layer_id]['replace_mode'] == 'insert_before':
                                inp = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    orig
                                )
                                tf_layers_dict[layer_id] = Conv2D(
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    dilation_rate=dilations,
                                    use_bias=False,
                                    kernel_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(2,3,1,0))
                                )(inp)

                            elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                                inp = Conv2D(
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    dilation_rate=dilations,
                                    use_bias=False,
                                    kernel_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(2,3,1,0))
                                )(orig)
                                tf_layers_dict[layer_id] = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    inp
                                )

                        else:
                            tf_layers_dict[layer_id] = Conv2D(
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                dilation_rate=dilations,
                                use_bias=False,
                                kernel_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(2,3,1,0))
                            )(orig)

                    except:
                        try:
                            if wr_config and layer_id in wr_config and format_version >= 2:
                                if wr_config[layer_id]['replace_mode'] == 'insert_before':
                                    inp = extrapolation_of_layers(
                                        wr_config[layer_id],
                                        orig
                                    )
                                    tf_layers_dict[layer_id] = Conv2D(
                                        filters=filters,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        dilation_rate=dilations,
                                        use_bias=False,
                                        kernel_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].numpy().transpose(2,3,1,0))
                                    )(inp)
                                elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                                    inp = Conv2D(
                                        filters=filters,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        dilation_rate=dilations,
                                        use_bias=False,
                                        kernel_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].numpy().transpose(2,3,1,0))
                                    )(orig)
                                    tf_layers_dict[layer_id] = extrapolation_of_layers(
                                        wr_config[layer_id],
                                        inp
                                    )
                            else:
                                tf_layers_dict[layer_id] = Conv2D(
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    dilation_rate=dilations,
                                    use_bias=False,
                                    kernel_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].numpy().transpose(2,3,1,0))
                                )(orig)

                        except:
                            # Weights from OP that are not fixed values
                            if wr_config and layer_id in wr_config and format_version >= 2:
                                if wr_config[layer_id]['replace_mode'] == 'insert_before':
                                    inp = extrapolation_of_layers(
                                        wr_config[layer_id],
                                        orig
                                    )
                                    tf_layers_dict[layer_id] = tf.nn.conv2d(
                                        input=inp,
                                        filters=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                                        strides=strides,
                                        padding=padding.upper(),
                                        dilations=dilations
                                    )

                                elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                                    inp = tf.nn.conv2d(
                                        input=orig,
                                        filters=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                                        strides=strides,
                                        padding=padding.upper(),
                                        dilations=dilations
                                    )
                                    tf_layers_dict[layer_id] = extrapolation_of_layers(
                                        wr_config[layer_id],
                                        inp
                                    )
                            else:
                                tf_layers_dict[layer_id] = tf.nn.conv2d(
                                    input=orig,
                                    filters=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                                    strides=strides,
                                    padding=padding.upper(),
                                    dilations=dilations
                                )

                elif len(strides) == 3:
                    # Conv3D - WIP padding same only
                    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3D
                    """
                    openvino = [C_OUT, C_IN, Z,    Y,     X] = [8,1,3,3,3] [16,8,3,3,3]
                    tf       = [    Z,    Y, X, C_IN, C_OUT] = [3,3,3,1,8] [3,3,3,8,16]
                    """
                    kernel_size = [int(port1[2]), int(port1[3]), int(port1[4])]
                    try:
                        if wr_config and layer_id in wr_config and format_version >= 2:
                            if wr_config[layer_id]['replace_mode'] == 'insert_before':
                                inp = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                                )
                                tf_layers_dict[layer_id] = Conv3D(
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding='same',
                                    dilation_rate=dilations,
                                    use_bias=False,
                                    kernel_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose((2,3,4,1,0)))
                                )(inp)
                            elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                                inp = Conv3D(
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding='same',
                                    dilation_rate=dilations,
                                    use_bias=False,
                                    kernel_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose((2,3,4,1,0)))
                                )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                tf_layers_dict[layer_id] = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    inp
                                )
                        else:
                            tf_layers_dict[layer_id] = Conv3D(
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding='same',
                                dilation_rate=dilations,
                                use_bias=False,
                                kernel_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose((2,3,4,1,0)))
                            )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

                    except:
                        # Weights from OP that are not fixed values
                        # https://www.tensorflow.org/api_docs/python/tf/nn/conv3d
                        strides = [1, strides[0], strides[1], strides[2], 1]
                        if wr_config and layer_id in wr_config and format_version >= 2:
                            if wr_config[layer_id]['replace_mode'] == 'insert_before':
                                inp = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                                )
                                tf_layers_dict[layer_id] = tf.nn.conv3d(
                                    input=inp,
                                    filters=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                                    strides=strides,
                                    padding="SAME",
                                    dilations=[1, dilations[0], dilations[1], dilations[2], 1]
                                )

                            elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                                inp = tf.nn.conv3d(
                                    input=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                    filters=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                                    strides=strides,
                                    padding="SAME",
                                    dilations=[1, dilations[0], dilations[1], dilations[2], 1]
                                )
                                tf_layers_dict[layer_id] = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    inp
                                )
                        else:
                            tf_layers_dict[layer_id] = tf.nn.conv3d(
                                input=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                filters=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                                strides=strides,
                                padding="SAME",
                                dilations=[1, dilations[0], dilations[1], dilations[2], 1]
                            )

                elif len(strides) == 1:
                    # Conv1D
                    """
                    VINO:[N, C, W]
                    TF  :[N, W, C]
                    VINO = input:[1,1024,16] filter:[512,1024,1]
                    TF   = input:[1,16,1024] filter:[1,1024,512]
                    """
                    kernel_size = [int(port1[2])]
                    if wr_config and layer_id in wr_config and format_version >= 2:
                        if wr_config[layer_id]['replace_mode'] == 'insert_before':
                            inp = extrapolation_of_layers(
                                wr_config[layer_id],
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                            )
                            tf_layers_dict[layer_id] = tf.nn.conv1d(
                                input=inp,
                                filters=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(2,1,0),
                                stride=strides,
                                padding=padding.upper(),
                                dilations=dilations
                            )

                        elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                            inp = tf.nn.conv1d(
                                input=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                filters=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(2,1,0),
                                stride=strides,
                                padding=padding.upper(),
                                dilations=dilations
                            )
                            tf_layers_dict[layer_id] = extrapolation_of_layers(
                                wr_config[layer_id],
                                inp
                            )
                    else:
                        tf_layers_dict[layer_id] = tf.nn.conv1d(
                            input=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            filters=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(2,1,0),
                            stride=strides,
                            padding=padding.upper(),
                            dilations=dilations
                        )


            ### Add
            elif layer.attrib['type'] == 'Add':
                # 'Fused_Add_' == BiasAdd
                if len(tf_edges[layer_id]) == 2 and type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]) == np.ndarray:
                    try:
                        # Biasadd or Add
                        edge_id0 = get_tf_edges_from(tf_edges, layer_id, 0)
                        edge_id1 = get_tf_edges_from(tf_edges, layer_id, 1)
                        if wr_config and layer_id in wr_config and format_version >= 2:
                            if wr_config[layer_id]['replace_mode'] == 'insert_before':
                                print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                                sys.exit(-1)

                            elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                                if tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape[-1] == tf_layers_dict[edge_id1].flatten().shape[0]:
                                    inp = tf.math.add(tf_layers_dict[edge_id0], tf_layers_dict[edge_id1].flatten())
                                else:
                                    inp = tf.math.add(tf_layers_dict[edge_id0], tf_layers_dict[edge_id1])
                                tf_layers_dict[layer_id] = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    inp
                                )
                        else:
                            if tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape[-1] == tf_layers_dict[edge_id1].flatten().shape[0]:
                                tf_layers_dict[layer_id] = tf.math.add(tf_layers_dict[edge_id0], tf_layers_dict[edge_id1].flatten())
                            else:
                                tf_layers_dict[layer_id] = tf.math.add(tf_layers_dict[edge_id0], tf_layers_dict[edge_id1])

                    except:
                        # Add
                        edge_id0 = get_tf_edges_from(tf_edges, layer_id, 0)
                        edge_id1 = get_tf_edges_from(tf_edges, layer_id, 1)
                        if wr_config and layer_id in wr_config and format_version >= 2:
                            if wr_config[layer_id]['replace_mode'] == 'insert_before':
                                print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                                sys.exit(-1)

                            elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                                try:
                                    inp = tf.math.add(tf_layers_dict[edge_id0], tf_layers_dict[edge_id1])
                                except:
                                    inp = tf.math.add(tf_layers_dict[edge_id0], tf_layers_dict[edge_id1].transpose(0,2,3,1))
                                tf_layers_dict[layer_id] = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    inp
                                )
                        else:
                            try:
                                tf_layers_dict[layer_id] = tf.math.add(tf_layers_dict[edge_id0], tf_layers_dict[edge_id1])
                            except:
                                tf_layers_dict[layer_id] = tf.math.add(tf_layers_dict[edge_id0], tf_layers_dict[edge_id1].transpose(0,2,3,1))

                else:
                    # Add
                    if wr_config and layer_id in wr_config and format_version >= 2:
                        if wr_config[layer_id]['replace_mode'] == 'insert_before':
                            print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                            sys.exit(-1)

                        elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                            if len(get_tf_edges_from(tf_edges, layer_id)) == 2:
                                try:
                                    tmp_layers = [tf_layers_dict[from_layer_id].transpose(0,2,3,1).astype(np.float32) if type(tf_layers_dict[from_layer_id]) == np.ndarray else tf_layers_dict[from_layer_id] for from_layer_id in get_tf_edges_from(tf_edges, layer_id)]
                                    inp = tf.math.add(tmp_layers[0], tmp_layers[1])

                                except:
                                    try:
                                        tmp_layers = [tf_layers_dict[from_layer_id].transpose(0,2,3,1) if type(tf_layers_dict[from_layer_id]) == np.ndarray else tf_layers_dict[from_layer_id] for from_layer_id in get_tf_edges_from(tf_edges, layer_id)]
                                        inp = tf.math.add(tmp_layers[0], tmp_layers[1])

                                    except:
                                        tmp_layers = [tf_layers_dict[from_layer_id] if type(tf_layers_dict[from_layer_id]) == np.ndarray else tf_layers_dict[from_layer_id] for from_layer_id in get_tf_edges_from(tf_edges, layer_id)]
                                        inp = tf.math.add(tmp_layers[0], tmp_layers[1])

                            else:
                                inp = tf.math.add_n(
                                    [tf_layers_dict[from_layer_id].transpose(0,2,3,1).astype(np.float32) if type(tf_layers_dict[from_layer_id]) == np.ndarray else tf_layers_dict[from_layer_id] for from_layer_id in get_tf_edges_from(tf_edges, layer_id)]
                                )
                            tf_layers_dict[layer_id] = extrapolation_of_layers(
                                wr_config[layer_id],
                                inp
                            )
                    else:
                        if len(get_tf_edges_from(tf_edges, layer_id)) == 2:
                            try:
                                tmp_layers = [tf_layers_dict[from_layer_id].transpose(0,2,3,1).astype(np.float32) if type(tf_layers_dict[from_layer_id]) == np.ndarray else tf_layers_dict[from_layer_id] for from_layer_id in get_tf_edges_from(tf_edges, layer_id)]
                                tf_layers_dict[layer_id] = tf.math.add(tmp_layers[0], tmp_layers[1])

                            except:
                                try:
                                    tmp_layers = [tf_layers_dict[from_layer_id].transpose(0,2,3,1) if type(tf_layers_dict[from_layer_id]) == np.ndarray else tf_layers_dict[from_layer_id] for from_layer_id in get_tf_edges_from(tf_edges, layer_id)]
                                    tf_layers_dict[layer_id] = tf.math.add(tmp_layers[0], tmp_layers[1])

                                except:
                                    tmp_layers = [tf_layers_dict[from_layer_id] if type(tf_layers_dict[from_layer_id]) == np.ndarray else tf_layers_dict[from_layer_id] for from_layer_id in get_tf_edges_from(tf_edges, layer_id)]
                                    tf_layers_dict[layer_id] = tf.math.add(tmp_layers[0], tmp_layers[1])

                        else:
                            tf_layers_dict[layer_id] = tf.math.add_n(
                                [tf_layers_dict[from_layer_id].transpose(0,2,3,1).astype(np.float32) if type(tf_layers_dict[from_layer_id]) == np.ndarray else tf_layers_dict[from_layer_id] for from_layer_id in get_tf_edges_from(tf_edges, layer_id)]
                            )

            ### ReLU
            elif layer.attrib['type'] == 'ReLU':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.nn.relu(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.nn.relu(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.nn.relu(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### PReLU
            elif layer.attrib['type'] == 'PReLU':
                input_len = len(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape)
                alpha_len = len(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].shape)

                shared_axes = []
                if alpha_len < 4:
                    if input_len == 4:
                        shared_axes = [1, 2]
                    elif input_len == 3:
                        shared_axes = [1]
                    else:
                        shared_axes = None
                else:
                    shared_axes = None

                temp_alpha = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]

                if alpha_len > 1 and temp_alpha.size == 1:
                    temp_alpha = np.squeeze(temp_alpha)
                    alpha_len = 1

                if alpha_len == 4 and temp_alpha.shape[0] == 1 and temp_alpha.shape[2] == 1 and temp_alpha.shape[3] == 1:
                    shared_axes = [1, 2]

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['type'] == 'PReLU' and wr_config[layer_id]['replace_mode'] == 'change_shared_axes':
                        shared_axes = wr_config[layer_id]['values']

                if alpha_len == 4:
                    if replace_prelu_and_minmax:
                        if wr_config and layer_id in wr_config and format_version >= 2:
                            if wr_config[layer_id]['replace_mode'] == 'insert_before':
                                inp = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                                )
                                try:
                                    tf_layers_dict[layer_id] = \
                                        tf.maximum(0.0, inp) + \
                                            tf.minimum(0.0, temp_alpha.transpose(0,2,3,1) * inp)
                                except:
                                    tf_layers_dict[layer_id] = \
                                        tf.maximum(0.0, inp) + \
                                            tf.minimum(0.0, tf.transpose(temp_alpha, perm=[0,2,3,1]) * inp)

                            elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                                try:
                                    inp = tf.maximum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) + \
                                        tf.minimum(0.0, temp_alpha.transpose(0,2,3,1) * tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                except:
                                    inp = tf.maximum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) + \
                                        tf.minimum(0.0, tf.transpose(temp_alpha, perm=[0,2,3,1]) * tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                tf_layers_dict[layer_id] = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    inp
                                )
                            else:
                                try:
                                    tf_layers_dict[layer_id] = \
                                        tf.maximum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) + \
                                            tf.minimum(0.0, temp_alpha.transpose(0,2,3,1) * tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                except:
                                    tf_layers_dict[layer_id] = \
                                        tf.maximum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) + \
                                            tf.minimum(0.0, tf.transpose(temp_alpha, perm=[0,2,3,1]) * tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

                        else:
                            try:
                                tf_layers_dict[layer_id] = \
                                    tf.maximum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) + \
                                        tf.minimum(0.0, temp_alpha.transpose(0,2,3,1) * tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                            except:
                                tf_layers_dict[layer_id] = \
                                    tf.maximum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) + \
                                        tf.minimum(0.0, tf.transpose(temp_alpha, perm=[0,2,3,1]) * tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

                    else:
                        if wr_config and layer_id in wr_config and format_version >= 2:
                            if wr_config[layer_id]['replace_mode'] == 'insert_before':
                                inp = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                                )
                                try:
                                    tf_layers_dict[layer_id] = PReLU(
                                        alpha_initializer=Constant(temp_alpha.transpose(0,2,3,1)),
                                        shared_axes=shared_axes
                                    )(inp)
                                except:
                                    try:
                                        tf_layers_dict[layer_id] = PReLU(
                                            alpha_initializer=Constant(tf.transpose(temp_alpha, perm=[0,2,3,1])),
                                            shared_axes=shared_axes
                                        )(inp)
                                    except:
                                        tf_layers_dict[layer_id] = PReLU(
                                            alpha_initializer=Constant(temp_alpha),
                                            shared_axes=shared_axes
                                        )(inp)

                            elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                                try:
                                    inp = PReLU(
                                        alpha_initializer=Constant(temp_alpha.transpose(0,2,3,1)),
                                        shared_axes=shared_axes
                                    )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                except:
                                    try:
                                        inp = PReLU(
                                            alpha_initializer=Constant(tf.transpose(temp_alpha, perm=[0,2,3,1])),
                                            shared_axes=shared_axes
                                        )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                    except:
                                        inp = PReLU(
                                            alpha_initializer=Constant(temp_alpha),
                                            shared_axes=shared_axes
                                        )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                tf_layers_dict[layer_id] = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    inp
                                )
                            else:
                                try:
                                    tf_layers_dict[layer_id] = PReLU(
                                        alpha_initializer=Constant(temp_alpha.transpose(0,2,3,1)),
                                        shared_axes=shared_axes
                                    )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                except:
                                    try:
                                        tf_layers_dict[layer_id] = PReLU(
                                            alpha_initializer=Constant(tf.transpose(temp_alpha, perm=[0,2,3,1])),
                                            shared_axes=shared_axes
                                        )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                    except:
                                        tf_layers_dict[layer_id] = PReLU(
                                            alpha_initializer=Constant(temp_alpha),
                                            shared_axes=shared_axes
                                        )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

                        else:
                            try:
                                tf_layers_dict[layer_id] = PReLU(
                                    alpha_initializer=Constant(temp_alpha.transpose(0,2,3,1)),
                                    shared_axes=shared_axes
                                )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                            except:
                                try:
                                    tf_layers_dict[layer_id] = PReLU(
                                        alpha_initializer=Constant(tf.transpose(temp_alpha, perm=[0,2,3,1])),
                                        shared_axes=shared_axes
                                    )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                except:
                                    tf_layers_dict[layer_id] = PReLU(
                                        alpha_initializer=Constant(temp_alpha),
                                        shared_axes=shared_axes
                                    )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

                elif alpha_len == 3:
                    if replace_prelu_and_minmax:
                        if wr_config and layer_id in wr_config and format_version >= 2:
                            if wr_config[layer_id]['replace_mode'] == 'insert_before':
                                inp = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                                )
                                if temp_alpha.shape[1] == 1 and temp_alpha.shape[2] == 1:
                                    tf_layers_dict[layer_id] = \
                                        tf.maximum(0.0, inp) + \
                                            tf.minimum(0.0, temp_alpha.transpose(1, 2, 0) * inp)
                                else:
                                    tf_layers_dict[layer_id] = \
                                        tf.maximum(0.0, inp) + \
                                            tf.minimum(0.0, temp_alpha * inp)

                            elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                                if temp_alpha.shape[1] == 1 and temp_alpha.shape[2] == 1:
                                    inp = tf.maximum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) + \
                                        tf.minimum(0.0, temp_alpha.transpose(1, 2, 0) * tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                else:
                                    inp = tf.maximum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) + \
                                        tf.minimum(0.0, temp_alpha * tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                tf_layers_dict[layer_id] = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    inp
                                )
                        else:
                            if temp_alpha.shape[1] == 1 and temp_alpha.shape[2] == 1:
                                tf_layers_dict[layer_id] = \
                                    tf.maximum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) + \
                                        tf.minimum(0.0, temp_alpha.transpose(1, 2, 0) * tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                            else:
                                tf_layers_dict[layer_id] = \
                                    tf.maximum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) + \
                                        tf.minimum(0.0, temp_alpha * tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

                    else:
                        if wr_config and layer_id in wr_config and format_version >= 2:
                            if wr_config[layer_id]['replace_mode'] == 'insert_before':
                                inp = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                                )
                                if temp_alpha.shape[1] == 1 and temp_alpha.shape[2] == 1:
                                    tf_layers_dict[layer_id] = PReLU(
                                        alpha_initializer=Constant(temp_alpha.transpose(1, 2, 0)),
                                        shared_axes=shared_axes
                                    )(inp)
                                else:
                                    tf_layers_dict[layer_id] = PReLU(
                                        alpha_initializer=Constant(temp_alpha),
                                        shared_axes=shared_axes
                                    )(inp)

                            elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                                if temp_alpha.shape[1] == 1 and temp_alpha.shape[2] == 1:
                                    inp = PReLU(
                                        alpha_initializer=Constant(temp_alpha.transpose(1, 2, 0)),
                                        shared_axes=shared_axes
                                    )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                else:
                                    inp = PReLU(
                                        alpha_initializer=Constant(temp_alpha),
                                        shared_axes=shared_axes
                                    )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                tf_layers_dict[layer_id] = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    inp
                                )
                            else:
                                if temp_alpha.shape[1] == 1 and temp_alpha.shape[2] == 1:
                                    tf_layers_dict[layer_id] = PReLU(
                                        alpha_initializer=Constant(temp_alpha.transpose(1, 2, 0)),
                                        shared_axes=shared_axes
                                    )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                else:
                                    tf_layers_dict[layer_id] = PReLU(
                                        alpha_initializer=Constant(temp_alpha),
                                        shared_axes=shared_axes
                                    )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

                        else:
                            try:
                                if temp_alpha.shape[1] == 1 and temp_alpha.shape[2] == 1:
                                    tf_layers_dict[layer_id] = PReLU(
                                        alpha_initializer=Constant(temp_alpha.transpose(1, 2, 0)),
                                        shared_axes=shared_axes
                                    )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                else:
                                    tf_layers_dict[layer_id] = PReLU(
                                        alpha_initializer=Constant(temp_alpha),
                                        shared_axes=shared_axes
                                    )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                            except:
                                tf_layers_dict[layer_id] = PReLU(
                                    alpha_initializer=Constant(temp_alpha),
                                    shared_axes=shared_axes
                                )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

                elif alpha_len == 1:
                    # alpha_len == 1 (LeakyReLU)
                    if replace_prelu_and_minmax:
                        if wr_config and layer_id in wr_config and format_version >= 2:
                            if wr_config[layer_id]['replace_mode'] == 'insert_before':
                                inp = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                                )
                                tf_layers_dict[layer_id] = \
                                    tf.maximum(0.0, inp) + \
                                        tf.minimum(0.0, temp_alpha * inp)

                            elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                                inp = tf.maximum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) + \
                                    tf.minimum(0.0, temp_alpha * tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                tf_layers_dict[layer_id] = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    inp
                                )
                        else:
                            tf_layers_dict[layer_id] = \
                                tf.maximum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) + \
                                    tf.minimum(0.0, temp_alpha * tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

                    elif output_edgetpu: # LeakyReLU -> Max/Min
                        if wr_config and layer_id in wr_config and format_version >= 2:
                            if wr_config[layer_id]['replace_mode'] == 'insert_before':
                                inp = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                                )
                                tf_layers_dict[layer_id] = tf.maximum(0.0, inp) + tf.minimum(0.0, inp * temp_alpha)

                            elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                                inp = tf.maximum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) + \
                                    tf.minimum(0.0, temp_alpha * tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                tf_layers_dict[layer_id] = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    inp
                                )
                            else:
                                tf_layers_dict[layer_id] = tf.maximum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) + \
                                    tf.minimum(0.0, temp_alpha * tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

                        else:
                            tf_layers_dict[layer_id] = \
                                tf.maximum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) + \
                                    tf.minimum(0.0, temp_alpha * tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

                    else:
                        if wr_config and layer_id in wr_config and format_version >= 2:
                            if wr_config[layer_id]['replace_mode'] == 'insert_before':
                                inp = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                                )
                                try:
                                    tf_layers_dict[layer_id] = LeakyReLU(
                                        alpha=temp_alpha
                                    )(inp)
                                except:
                                    tf_layers_dict[layer_id] = PReLU(
                                        alpha_initializer=Constant(temp_alpha),
                                        shared_axes=shared_axes
                                    )(inp)

                            elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                                try:
                                    inp = LeakyReLU(
                                        alpha=temp_alpha
                                    )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                except:
                                    inp = PReLU(
                                        alpha_initializer=Constant(temp_alpha),
                                        shared_axes=shared_axes
                                    )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                tf_layers_dict[layer_id] = extrapolation_of_layers(
                                    wr_config[layer_id],
                                    inp
                                )
                            else:
                                tf_layers_dict[layer_id] = PReLU(
                                    alpha_initializer=Constant(temp_alpha),
                                    shared_axes=shared_axes
                                )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

                        else:
                            try:
                                tf_layers_dict[layer_id] = LeakyReLU(
                                        alpha=temp_alpha
                                    )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                            except:
                                tf_layers_dict[layer_id] = PReLU(
                                    alpha_initializer=Constant(temp_alpha),
                                    shared_axes=shared_axes
                                )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

                else:
                    print(f'{Color.RED}ERROR:{Color.RESET} Unsupported PReLU parameter dimension of {alpha_len}. layer_id: {layer_id}')
                    sys.exit(-1)

            ### Clamp
            elif layer.attrib['type'] == 'Clamp':
                cmin = None
                cmax = None
                if tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].dtype == np.int64:
                    cmin = np.asarray(data.attrib['min']).astype(np.int64)
                    cmax = np.asarray(data.attrib['max']).astype(np.int64)
                elif tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].dtype == np.int32:
                    cmin = np.asarray(data.attrib['min']).astype(np.int32)
                    cmax = np.asarray(data.attrib['max']).astype(np.int32)
                elif tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].dtype == np.float32:
                    cmin = np.asarray(data.attrib['min']).astype(np.float32)
                    cmax = np.asarray(data.attrib['max']).astype(np.float32)
                else:
                    cmin = np.asarray(data.attrib['min']).astype(np.float32)
                    cmax = np.asarray(data.attrib['max']).astype(np.float32)

                if cmin == 0.0 and cmax == 6.0:
                    # ReLU6
                    if wr_config and layer_id in wr_config and format_version >= 2:
                        if wr_config[layer_id]['replace_mode'] == 'insert_before':
                            inp = extrapolation_of_layers(
                                wr_config[layer_id],
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                            )
                            tf_layers_dict[layer_id] = tf.nn.relu6(inp)

                        elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                            inp = tf.nn.relu6(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                            tf_layers_dict[layer_id] = extrapolation_of_layers(
                                wr_config[layer_id],
                                inp
                            )
                    else:
                        tf_layers_dict[layer_id] = tf.nn.relu6(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

                else:
                    # Other
                    if wr_config and layer_id in wr_config and format_version >= 2:
                        if wr_config[layer_id]['replace_mode'] == 'insert_before':
                            inp = extrapolation_of_layers(
                                wr_config[layer_id],
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                            )
                            tf_layers_dict[layer_id] = tf.clip_by_value(
                                inp,
                                clip_value_min=cmin,
                                clip_value_max=cmax
                            )

                        elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                            inp = tf.clip_by_value(
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                clip_value_min=cmin,
                                clip_value_max=cmax
                            )
                            tf_layers_dict[layer_id] = extrapolation_of_layers(
                                wr_config[layer_id],
                                inp
                            )
                    else:
                        tf_layers_dict[layer_id] = tf.clip_by_value(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            clip_value_min=cmin,
                            clip_value_max=cmax
                        )

            ### Tan
            elif layer.attrib['type'] == 'Tan':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.math.tan(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.math.tan(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.math.tan(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Tanh
            elif layer.attrib['type'] == 'Tanh':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.math.tanh(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.math.tanh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.math.tanh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Elu
            elif layer.attrib['type'] == 'Elu':
                alpha = float(data.attrib['alpha'])
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = elu(inp, alpha=alpha)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = elu(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], alpha=alpha)
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = elu(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], alpha=alpha)

            ### HardSigmoid
            elif layer.attrib['type'] == 'HardSigmoid':
                x     = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                alpha = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                beta  = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            x
                        )
                        tf_layers_dict[layer_id] = tf.maximum([0.0], tf.minimum([1.0], tf.math.add(tf.math.multiply(alpha, inp), beta)))

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.maximum([0.0], tf.minimum([1.0], tf.math.add(tf.math.multiply(alpha, x), beta)))
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.maximum([0.0], tf.minimum([1.0], tf.math.add(tf.math.multiply(alpha, x), beta)))

            ### Sigmoid
            elif layer.attrib['type'] == 'Sigmoid':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.math.sigmoid(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.math.sigmoid(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.math.sigmoid(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### HSigmoid
            elif layer.attrib['type'] == 'HSigmoid':
                x = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            x
                        )
                        tf_layers_dict[layer_id] = \
                            tf.math.divide(tf.minimum(tf.maximum(tf.math.add(inp, [3.0]), [0.0]), [6.0]), [6.0])

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.math.divide(tf.minimum(tf.maximum(tf.math.add(x, [3.0]), [0.0]), [6.0]), [6.0])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = \
                        tf.math.divide(tf.minimum(tf.maximum(tf.math.add(x, [3.0]), [0.0]), [6.0]), [6.0])

            ### Swish
            elif layer.attrib['type'] == 'Swish':
                if replace_swish_and_hardswish:
                    # Hard-Swish
                    multiplier = 0.0
                    if not optimizing_hardswish_for_edgetpu:
                        multiplier = 0.16666667
                    else:
                        multiplier = 0.16666666

                    if wr_config and layer_id in wr_config and format_version >= 2:
                        if wr_config[layer_id]['replace_mode'] == 'insert_before':
                            inp = extrapolation_of_layers(
                                wr_config[layer_id],
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                            )
                            tf_layers_dict[layer_id] = inp * tf.nn.relu6(inp + 3) * multiplier

                        elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                            inp = \
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] * tf.nn.relu6(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] + 3) * multiplier
                            tf_layers_dict[layer_id] = extrapolation_of_layers(
                                wr_config[layer_id],
                                inp
                            )
                    else:
                        tf_layers_dict[layer_id] = \
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] * tf.nn.relu6(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] + 3) * multiplier

                else:
                    # Swish
                    if wr_config and layer_id in wr_config and format_version >= 2:
                        if wr_config[layer_id]['replace_mode'] == 'insert_before':
                            inp = extrapolation_of_layers(
                                wr_config[layer_id],
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                            )
                            tf_layers_dict[layer_id] = tf.nn.swish(inp)

                        elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                            inp = tf.nn.swish(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                            tf_layers_dict[layer_id] = extrapolation_of_layers(
                                wr_config[layer_id],
                                inp
                            )
                    else:
                        tf_layers_dict[layer_id] = tf.nn.swish(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])


            ### SoftPlus
            elif layer.attrib['type'] == 'SoftPlus':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.math.softplus(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.math.softplus(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.math.softplus(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### MaxPool
            elif layer.attrib['type'] == 'MaxPool':
                outport_size = sum([int(sdim.text) for sdim in layer.find('output')[0]])
                kernel_size =  [int(s) for s in data.attrib['kernel'].split(',')]
                strides = [int(s) for s in data.attrib['strides'].split(',')]
                pads_begin = 0
                pads_end = 0
                if not data is None and 'pads_begin' in data.attrib:
                    pads_begin = sum([int(s) for s in data.attrib['pads_begin'].split(',')])
                if not data is None and 'pads_end' in data.attrib:
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

                temp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]

                if pads_begin > 0:
                    padding = 'VALID'
                    # begin 0 = top
                    # begin 1 = left
                    # end 0 = bottom
                    # end 1 = right
                    begin = [int(data.attrib['pads_begin'].split(',')[0]), int(data.attrib['pads_end'].split(',')[0])]
                    end   = [int(data.attrib['pads_begin'].split(',')[1]), int(data.attrib['pads_end'].split(',')[1])]

                    changed_padding_mode = None
                    if wr_config and layer_id in wr_config and format_version >= 2:
                        if wr_config[layer_id]['type'] == 'MaxPool' and wr_config[layer_id]['replace_mode'] == 'change_padding_mode':
                            changed_padding_mode = wr_config[layer_id]['values']


                    if not changed_padding_mode:
                        if begin == [1,1] and end == [1,1]:
                            orig = tf.keras.layers.ZeroPadding2D([begin, end])(temp)
                        else:
                            # paddings = tf.constant([[0,0],[2,2],[2,2],[0,0]])
                            # D0: [before, after]
                            # D1: [before, after]
                            # D2: [before, after]
                            # D3: [before, after]
                            # [D0, D1, D2, D3]
                            pads = np.asarray([[0,0],begin,end,[0,0]], dtype=np.int32)
                            orig = tf.pad(
                                temp,
                                paddings=pads,
                                mode='SYMMETRIC'
                            )
                    else:
                        # changed_padding_mode in ['ZERO', 'SYMMETRIC', 'REFLECT']
                        if changed_padding_mode.upper() == 'ZERO':
                            orig = tf.keras.layers.ZeroPadding2D([begin, end])(temp)
                        else:
                            # paddings = tf.constant([[0,0],[2,2],[2,2],[0,0]])
                            # D0: [before, after]
                            # D1: [before, after]
                            # D2: [before, after]
                            # D3: [before, after]
                            # [D0, D1, D2, D3]
                            pads = np.asarray([[0,0],begin,end,[0,0]], dtype=np.int32)
                            orig = tf.pad(
                                temp,
                                paddings=pads,
                                mode=changed_padding_mode.upper()
                            )
                else:
                    if temp.shape[0] == 1 and temp.shape[2] == 1 and temp.shape[3] == 1:
                        orig = tf.transpose(temp, perm=(0,2,3,1))
                    else:
                        orig = temp

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            orig
                        )
                        tf_layers_dict[layer_id] = tf.nn.max_pool(
                            inp,
                            ksize=kernel_size,
                            strides=strides,
                            padding=padding
                        )

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.nn.max_pool(
                            orig,
                            ksize=kernel_size,
                            strides=strides,
                            padding=padding
                        )
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                    else:
                        tf_layers_dict[layer_id] = tf.nn.max_pool(
                            orig,
                            ksize=kernel_size,
                            strides=strides,
                            padding=padding
                        )

                else:
                    tf_layers_dict[layer_id] = tf.nn.max_pool(
                        orig,
                        ksize=kernel_size,
                        strides=strides,
                        padding=padding
                    )

                new_layer_outport_size = sum([sdim for sdim in tf_layers_dict[layer_id].shape])
                if outport_size != new_layer_outport_size:
                    # Caffe -> TF
                    if wr_config and layer_id in wr_config and format_version >= 2:
                        if wr_config[layer_id]['replace_mode'] == 'insert_before':
                            inp = extrapolation_of_layers(
                                wr_config[layer_id],
                                orig
                            )
                            tf_layers_dict[layer_id] = tf.nn.max_pool(
                                inp,
                                ksize=kernel_size,
                                strides=strides,
                                padding='SAME'
                            )

                        elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                            inp = tf.nn.max_pool(
                                orig,
                                ksize=kernel_size,
                                strides=strides,
                                padding='SAME'
                            )
                            tf_layers_dict[layer_id] = extrapolation_of_layers(
                                wr_config[layer_id],
                                inp
                            )

                    else:
                        tf_layers_dict[layer_id] = tf.nn.max_pool(
                            orig,
                            ksize=kernel_size,
                            strides=strides,
                            padding='SAME'
                        )

            ### AvgPool
            elif layer.attrib['type'] == 'AvgPool':
                kernel_size =  [int(s) for s in data.attrib['kernel'].split(',')]
                strides = [int(s) for s in data.attrib['strides'].split(',')]
                # exclude_pad = data.attrib['exclude-pad']
                pads_begin = 0
                pads_end = 0
                if not data is None and 'pads_begin' in data.attrib:
                    pads_begin = sum([int(s) for s in data.attrib['pads_begin'].split(',')])
                if not data is None and 'pads_end' in data.attrib:
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
                if not data is None and 'rounding_type' in data.attrib and data.attrib['rounding_type'] == 'ceil':
                    padding = 'same'

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = AveragePooling2D(
                            pool_size=kernel_size,
                            strides=strides,
                            padding=padding
                        )(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = AveragePooling2D(
                            pool_size=kernel_size,
                            strides=strides,
                            padding=padding
                        )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )

                else:
                    tf_layers_dict[layer_id] = AveragePooling2D(
                        pool_size=kernel_size,
                        strides=strides,
                        padding=padding
                    )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### GroupConvolution
            elif layer.attrib['type'] == 'GroupConvolution':
                port0 = [int(sdim.text) for sdim in layer.find('input')[0]]
                port1 = [int(sdim.text) for sdim in layer.find('input')[1]]
                depth_multiplier = 1
                kernel_size = [int(port1[3]), int(port1[4])]
                strides = [int(s) for s in data.attrib['strides'].split(',')]
                pads_begin = 0
                pads_end = 0
                if not data is None and 'pads_begin' in data.attrib:
                    pads_begin = sum([int(s) for s in data.attrib['pads_begin'].split(',')])
                if not data is None and 'pads_end' in data.attrib:
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

                temp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                orig = None
                if pads_begin > 0:
                    padding = 'valid'
                    # begin 0 = top
                    # begin 1 = left
                    # end 0 = bottom
                    # end 1 = right
                    begin = [int(data.attrib['pads_begin'].split(',')[0]), int(data.attrib['pads_end'].split(',')[0])]
                    end   = [int(data.attrib['pads_begin'].split(',')[1]), int(data.attrib['pads_end'].split(',')[1])]
                    orig = tf.keras.layers.ZeroPadding2D([begin, end])(temp)
                else:
                    orig = temp

                dilations = [int(s) for s in data.attrib['dilations'].split(',')]

                inp = None
                if int(port1[1]) > 1:
                    # Conv2D with groups
                    filters = int(port0[1])
                    groups = int(port1[0])

                    convs = []
                    kernel = None
                    if len(port1) == 5:
                        try:
                            kernel = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(3,4,2,1,0)
                        except:
                            kernel = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].numpy().transpose(3,4,2,1,0)
                        for i in range(groups):
                            convs.append(
                                Conv2D(
                                    filters=filters // groups,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    dilation_rate=dilations,
                                    use_bias=False,
                                    kernel_initializer=Constant(kernel[:,:,:,:,i])
                                )
                            )
                    else:
                        try:
                            kernel = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(2,3,1,0)
                        except:
                            kernel = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].numpy().transpose(2,3,1,0)
                        for i in range(groups):
                            convs.append(
                                Conv2D(
                                    filters=filters // groups,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    dilation_rate=dilations,
                                    use_bias=False,
                                    kernel_initializer=Constant(kernel[:,:,:,i])
                                )
                            )

                    try:
                        x_splits = tf.split(orig, groups, -1)
                        x_outputs = [conv(x_split) for x_split, conv in zip(x_splits, convs)]
                        inp = tf.concat(x_outputs, -1)
                    except Exception as e:
                        if len(port1) == 5:
                            kernel = None
                            temp_val = None
                            filters = None
                            try:
                                temp_val = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                                filters = temp_val.shape[0] * temp_val.shape[1]
                                temp_val = temp_val.reshape([filters, temp_val.shape[2], temp_val.shape[3], temp_val.shape[4]])
                                kernel = temp_val.transpose([2,3,1,0])
                            except:
                                temp_val = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].numpy()
                                filters = temp_val.shape[0] * temp_val.shape[1]
                                temp_val = temp_val.reshape([filters, temp_val.shape[2], temp_val.shape[3], temp_val.shape[4]])
                                kernel = temp_val.transpose([2,3,1,0])
                            inp = Conv2D(
                                filters=filters,
                                kernel_size=kernel_size,
                                padding=padding,
                                groups=groups,
                                use_bias=False,
                                kernel_initializer=Constant(kernel)
                            )(orig)

                        else:
                            raise e

                else:
                    # DepthwiseConv2D
                    try:
                        inp = DepthwiseConv2D(
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            depth_multiplier=depth_multiplier,
                            dilation_rate=dilations,
                            use_bias=False,
                            depthwise_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(3,4,1,2,0))
                        )(orig)
                    except:
                        inp = DepthwiseConv2D(
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            depth_multiplier=depth_multiplier,
                            dilation_rate=dilations,
                            use_bias=False,
                            depthwise_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].numpy().transpose(3,4,1,2,0))
                        )(orig)

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp


            ### ConvolutionBackpropData
            elif layer.attrib['type'] == 'ConvolutionBackpropData':
                # port0 = [int(sdim.text) for sdim in layer.find('input')[0]]
                port1 = [int(sdim.text) for sdim in layer.find('input')[1]]
                port2 = [int(sdim.text) for sdim in layer.find('output')[0]]
                filters = int(port2[1])
                kernel_size = [int(port1[2]), int(port1[3])]
                strides = [int(s) for s in data.attrib['strides'].split(',')]
                pads_begin = 0
                pads_end = 0
                if not data is None and 'pads_begin' in data.attrib:
                    pads_begin = sum([int(s) for s in data.attrib['pads_begin'].split(',')])
                if not data is None and 'pads_end' in data.attrib:
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

                if len(strides) == 2:
                    # Conv2DTranspose
                    if wr_config and layer_id in wr_config and format_version >= 2:
                        if wr_config[layer_id]['replace_mode'] == 'insert_before':
                            inp = extrapolation_of_layers(
                                wr_config[layer_id],
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                            )
                            tf_layers_dict[layer_id] = Conv2DTranspose(
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                dilation_rate=dilations,
                                use_bias=False,
                                kernel_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(2,3,1,0))
                            )(inp)

                        elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                            inp = Conv2DTranspose(
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                dilation_rate=dilations,
                                use_bias=False,
                                kernel_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(2,3,1,0))
                            )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                            tf_layers_dict[layer_id] = extrapolation_of_layers(
                                wr_config[layer_id],
                                inp
                            )

                    else:
                        tf_layers_dict[layer_id] = Conv2DTranspose(
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            dilation_rate=dilations,
                            use_bias=False,
                            kernel_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(2,3,1,0))
                        )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

                elif len(strides) == 3:
                    # Conv3DTranspose - WIP padding same only
                    # https://www.tensorflow.org/api_docs/python/tf/nn/conv3d_transpose
                    """
                    in =
                        [N, C_INPUT, Z, Y, X]

                    kernel =
                        [C_INPUT, C_OUTPUT, Z, Y, X]
                        [     48,       32, 4, 4, 4]

                        [N, Z, Y,  X, C_INPUT]
                        [1, 6, 8, 11,      48]
                        [1, 6, 8, 11,      48]

                        [Z,  Y,   X, C_OUTPUT, C_INPUT]
                        [4, 4, 4, 32, 48]

                        [N, C_OUTPUT,  Z,  Y,  X]
                        [1,       32, 12, 16, 22]
                        [1,       12, 16, 22, 32]
                    """
                    output_shape = np.asarray([int(v.text) for v in layer.find("output").find("port")])
                    if wr_config and layer_id in wr_config and format_version >= 2:
                        if wr_config[layer_id]['replace_mode'] == 'insert_before':
                            inp = extrapolation_of_layers(
                                wr_config[layer_id],
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                            )
                            tf_layers_dict[layer_id] = tf.nn.conv3d_transpose(
                                input=inp,
                                filters=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose((2,3,4,1,0)),
                                output_shape=output_shape[[0,2,3,4,1]],
                                strides=strides[0],
                                padding='SAME'
                            )

                        elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                            inp = tf.nn.conv3d_transpose(
                                input=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                filters=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose((2,3,4,1,0)),
                                output_shape=output_shape[[0,2,3,4,1]],
                                strides=strides[0],
                                padding='SAME'
                            )
                            tf_layers_dict[layer_id] = extrapolation_of_layers(
                                wr_config[layer_id],
                                inp
                            )
                    else:
                        tf_layers_dict[layer_id] = tf.nn.conv3d_transpose(
                            input=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            filters=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose((2,3,4,1,0)),
                            output_shape=output_shape[[0,2,3,4,1]],
                            strides=strides[0],
                            padding='SAME'
                        )

            ### Concat
            elif layer.attrib['type'] == 'Concat':
                axis = -1
                if 'axis' in data.attrib:
                    axis = int(data.attrib['axis'])
                if axis == 1 and len(np.asarray(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape)) == 4:
                    axis = -1
                elif (axis == -1 or axis == len(np.asarray(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape)) - 1) and \
                    len(np.asarray(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape)) == 4:
                    axis = 1
                elif len(np.asarray(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape)) < 4:
                    pass
                elif axis > 0:
                    pass

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['type'] == 'Concat' and wr_config[layer_id]['replace_mode'] == 'change_axis':
                        axis = int(wr_config[layer_id]['values'])

                length_list = []
                for from_layer_id in get_tf_edges_from(tf_edges, layer_id):
                    length_list.append(len(tf_layers_dict[from_layer_id].shape))
                axis_zero_count = length_list.count(0)
                sum_of_axis = sum(length_list)
                if axis_zero_count > 0 and sum_of_axis > 0:
                    tensor_list = []
                    for length, from_layer_id in zip(length_list, get_tf_edges_from(tf_edges, layer_id)):
                        if length == 0:
                            tensor_list.append([tf_layers_dict[from_layer_id]])
                        else:
                            tensor_list.append(tf_layers_dict[from_layer_id])
                    inp = tf.concat(tensor_list, axis=axis)
                else:
                    temp = get_tf_edges_from(tf_edges, layer_id)
                    inp = tf.concat(
                        [tf_layers_dict[from_layer_id] for from_layer_id in get_tf_edges_from(tf_edges, layer_id)],
                        axis=axis
                    )

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                    else:
                        tf_layers_dict[layer_id] = inp
                else:
                    tf_layers_dict[layer_id] = inp

            ### Multiply
            elif layer.attrib['type'] == 'Multiply':
                if len(tf_edges[layer_id]) == 2 and (type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]) == np.ndarray):
                    if tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].ndim == 4:
                        # 4D - NCHW->NHWC
                        inp = tf.math.multiply(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(0,2,3,1).astype(np.float32)
                        )
                    else:
                        # unknown
                        try:
                            x_shape = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].type_spec.shape
                            y_shape = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].shape
                            if x_shape == y_shape:
                                inp = tf.math.multiply(
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                                )
                            else:
                                try:
                                    inp = tf.math.multiply(
                                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].reshape(x_shape)
                                    )
                                except:
                                    try:
                                        inp = tf.math.multiply(
                                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(0,2,1)
                                        )
                                    except:
                                        inp = tf.math.multiply(
                                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                                        )
                        except:
                            inp = tf.math.multiply(
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                            )
                elif len(tf_edges[layer_id]) == 2 and (type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) == np.ndarray):
                    if tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].ndim == 4:
                        # 4D - NCHW->NHWC
                        inp = tf.math.multiply(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].transpose(0,2,3,1).astype(np.float32),
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                        )
                    else:
                        # unknown
                        try:
                            x_shape = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape
                            y_shape = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].type_spec.shape
                            if x_shape == y_shape:
                                inp = tf.math.multiply(
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                                )
                            else:
                                try:
                                    inp = tf.math.multiply(
                                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].reshape(y_shape),
                                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                                    )
                                except:
                                    try:
                                        inp = tf.math.multiply(
                                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].transpose(0,2,1),
                                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                                        )
                                    except:
                                        inp = tf.math.multiply(
                                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                                        )
                        except:
                            inp = tf.math.multiply(
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                            )
                else:
                    # unknown
                    try:
                        inp = tf.math.multiply(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                        )
                    except:
                        # squaring
                        inp = tf.math.square(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### Interpolate
            elif layer.attrib['type'] == 'Interpolate':
                mode = data.attrib['mode']
                if mode == 'linear_onnx':
                    mode = 'linear'
                antialias = False
                try:
                    antialias = False if int(data.attrib['antialias']) == 0 else True
                except:
                    antialias = False if data.attrib['antialias'].upper() == 'FALSE' else True
                input_shape = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape
                out_port0 = [int(sdim.text) for sdim in layer.find('output')[0]]

                def upsampling2d_bilinear(x, upsampling_factor_height, upsampling_factor_width):
                    h = x.shape[1] * upsampling_factor_height
                    w = x.shape[2] * upsampling_factor_width
                    if output_edgetpu:
                        if verbose:
                            print(f'{Color.YELLOW}WARNING:{Color.RESET} The weights after Upsampling (tf.compat.v1.image.resize_bilinear) are shifted to the upper left. If you do not need to generate EdgeTPU models, set --output_edgetpu False and run again. OP: {x.name}')
                        return tf.compat.v1.image.resize_bilinear(x, (h, w))
                    elif output_coreml:
                        return tf.compat.v1.image.resize_bilinear(x, (h, w), align_corners=True)
                    else:
                        return tf.image.resize(x, [h, w], method='bilinear')

                def upsampling2d_nearest(x, upsampling_factor_height, upsampling_factor_width):
                    h = x.shape[1] * upsampling_factor_height
                    w = x.shape[2] * upsampling_factor_width
                    if output_edgetpu:
                        if verbose:
                            print(f'{Color.YELLOW}WARNING:{Color.RESET} The weights after Upsampling (tf.compat.v1.image.resize_nearest_neighbor) are shifted to the upper left. If you do not need to generate EdgeTPU models, set --output_edgetpu False and run again. OP: {x.name}')
                        return tf.compat.v1.image.resize_nearest_neighbor(x, (h, w))
                    elif output_coreml:
                        return tf.compat.v1.image.resize_nearest_neighbor(x, (h, w))
                    else:
                        return tf.image.resize(x, [h, w], method='nearest')

                def upsampling2d_bilinear_5d(x, upsampling_factor_depth, upsampling_factor_height, upsampling_factor_width):
                    d = x.shape[1] * upsampling_factor_depth
                    h = x.shape[2] * upsampling_factor_height
                    w = x.shape[3] * upsampling_factor_width
                    # Dpeth (height x width)
                    resized_list = []
                    unstack_img_list = tf.unstack(x, axis=1)
                    for i in unstack_img_list:
                        if output_edgetpu:
                            if verbose:
                                print(f'{Color.YELLOW}WARNING:{Color.RESET} The weights after Upsampling (tf.compat.v1.image.resize_nearest_neighbor) are shifted to the upper left. If you do not need to generate EdgeTPU models, set --output_edgetpu False and run again. OP: {x.name}')
                            resized_list.append(tf.compat.v1.image.resize_bilinear(x, (h, w)))
                        elif output_coreml:
                            resized_list.append(tf.compat.v1.image.resize_bilinear(x, (h, w), align_corners=True))
                        else:
                            resized_list.append(tf.image.resize(i, [h, w], method='bilinear'))
                    stack_img_hw = tf.stack(resized_list, axis=1)
                    # Width (depth x Height)
                    resized_list = []
                    unstack_img_list = tf.unstack(stack_img_hw, axis=3)
                    for i in unstack_img_list:
                        if output_edgetpu:
                            if verbose:
                                print(f'{Color.YELLOW}WARNING:{Color.RESET} The weights after Upsampling (tf.compat.v1.image.resize_nearest_neighbor) are shifted to the upper left. If you do not need to generate EdgeTPU models, set --output_edgetpu False and run again. OP: {x.name}')
                            resized_list.append(tf.compat.v1.image.resize_bilinear(x, (d, h)))
                        elif output_coreml:
                            resized_list.append(tf.compat.v1.image.resize_bilinear(x, (d, h), align_corners=True))
                        else:
                            resized_list.append(tf.image.resize(i, [d, h], method='bilinear'))
                    stack_img_dh = tf.stack(resized_list, axis=3)
                    return stack_img_dh

                def upsampling2d_nearest_5d(x, upsampling_factor_depth, upsampling_factor_height, upsampling_factor_width):
                    d = x.shape[1] * upsampling_factor_depth
                    h = x.shape[2] * upsampling_factor_height
                    w = x.shape[3] * upsampling_factor_width
                    # Dpeth (height x width)
                    resized_list = []
                    unstack_img_list = tf.unstack(x, axis=1)
                    for i in unstack_img_list:
                        if output_edgetpu:
                            if verbose:
                                print(f'{Color.YELLOW}WARNING:{Color.RESET} The weights after Upsampling (tf.compat.v1.image.resize_nearest_neighbor) are shifted to the upper left. If you do not need to generate EdgeTPU models, set --output_edgetpu False and run again. OP: {x.name}')
                            resized_list.append(tf.compat.v1.image.resize_nearest_neighbor(x, (h, w)))
                        elif output_coreml:
                            resized_list.append(tf.compat.v1.image.resize_nearest_neighbor(x, (h, w)))
                        else:
                            resized_list.append(tf.image.resize(i, [h, w], method='nearest'))
                    stack_img_hw = tf.stack(resized_list, axis=1)
                    # Width (depth x Height)
                    resized_list = []
                    unstack_img_list = tf.unstack(stack_img_hw, axis=3)
                    for i in unstack_img_list:
                        if output_edgetpu:
                            if verbose:
                                print(f'{Color.YELLOW}WARNING:{Color.RESET} The weights after Upsampling (tf.compat.v1.image.resize_nearest_neighbor) are shifted to the upper left. If you do not need to generate EdgeTPU models, set --output_edgetpu False and run again. OP: {x.name}')
                            resized_list.append(tf.compat.v1.image.resize_nearest_neighbor(x, (d, h)))
                        elif output_coreml:
                            resized_list.append(tf.compat.v1.image.resize_nearest_neighbor(x, (d, h)))
                        else:
                            resized_list.append(tf.image.resize(i, [d, h], method='nearest'))
                    stack_img_dh = tf.stack(resized_list, axis=3)
                    return stack_img_dh

                inp = None
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                else:
                    inp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]

                if len(input_shape) == 4:
                    # 4D Tensor N,H,W,C
                    out_height = int(out_port0[2])
                    out_width  = int(out_port0[3])

                    input_shape_height = input_shape[1]
                    input_shape_width  = input_shape[2]
                    upsampling_factor_height = out_height // input_shape_height
                    upsampling_factor_width  = out_width  // input_shape_width

                    if not restricted_resize_image_mode:

                        if (upsampling_factor_height * input_shape_height) == out_height and \
                            (upsampling_factor_width * input_shape_width) == out_width and \
                                upsampling_factor_height >= 1.0 and \
                                    upsampling_factor_width >= 1.0:
                            # Upsampling
                            if mode == 'linear':
                                inp = Lambda(
                                    upsampling2d_bilinear,
                                    arguments={
                                        'upsampling_factor_height': upsampling_factor_height,
                                        'upsampling_factor_width': upsampling_factor_width
                                    }
                                )(inp)
                            elif mode == 'nearest':
                                inp = Lambda(
                                    upsampling2d_nearest,
                                    arguments={
                                        'upsampling_factor_height': upsampling_factor_height,
                                        'upsampling_factor_width': upsampling_factor_width
                                    }
                                )(inp)
                            else:
                                print(f'The Interpolate - {mode} is not yet implemented.')
                                sys.exit(-1)

                        else:
                            # Others
                            if mode == 'linear':
                                if output_edgetpu:
                                    inp = tf.compat.v1.image.resize(
                                        inp,
                                        [out_height, out_width],
                                        method='bilinear'
                                    )
                                else:
                                    inp = tf.image.resize(
                                        inp,
                                        [out_height, out_width],
                                        method='bilinear'
                                    )
                            elif mode == 'nearest':
                                if output_edgetpu:
                                    inp = tf.compat.v1.image.resize(
                                        inp,
                                        [out_height, out_width],
                                        method='nearest'
                                    )
                                else:
                                    inp = tf.image.resize(
                                        inp,
                                        [out_height, out_width],
                                        method='nearest'
                                    )
                            else:
                                print(f'The Interpolate - {mode} is not yet implemented.')
                                sys.exit(-1)
                    else:
                        if mode == 'linear':
                            inp = tf.image.resize(
                                inp,
                                [out_height, out_width],
                                method='bilinear'
                            )
                        elif mode == 'nearest':
                            inp = tf.image.resize(
                                inp,
                                [out_height, out_width],
                                method='nearest'
                            )
                        else:
                            print(f'The Interpolate - {mode} is not yet implemented.')
                            sys.exit(-1)

                    if wr_config and layer_id in wr_config and format_version >= 2:
                        if wr_config[layer_id]['replace_mode'] == 'insert_before':
                            tf_layers_dict[layer_id] = inp

                        elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                            tf_layers_dict[layer_id] = extrapolation_of_layers(
                                wr_config[layer_id],
                                inp
                            )
                    else:
                        tf_layers_dict[layer_id] = inp

                elif len(input_shape) == 5:
                    # 5D Tensor N,D,H,W,C
                    out_depth = int(out_port0[2])
                    out_height = int(out_port0[3])
                    out_width  = int(out_port0[4])

                    input_shape_depth  = input_shape[1]
                    input_shape_height = input_shape[2]
                    input_shape_width  = input_shape[3]
                    upsampling_factor_depth  = out_depth  // input_shape_depth
                    upsampling_factor_height = out_height // input_shape_height
                    upsampling_factor_width  = out_width  // input_shape_width

                    if not restricted_resize_image_mode:

                        if (upsampling_factor_depth * input_shape_depth) == out_depth and \
                            (upsampling_factor_height * input_shape_height) == out_height and \
                            (upsampling_factor_width * input_shape_width) == out_width and \
                                upsampling_factor_depth >= 1.0 and \
                                    upsampling_factor_height >= 1.0 and \
                                        upsampling_factor_width >= 1.0:
                            # Upsampling
                            if mode == 'linear':
                                inp = Lambda(
                                    upsampling2d_bilinear_5d,
                                    arguments={
                                        'upsampling_factor_depth': upsampling_factor_depth,
                                        'upsampling_factor_height': upsampling_factor_height,
                                        'upsampling_factor_width': upsampling_factor_width
                                    }
                                )(inp)
                            elif mode == 'nearest':
                                inp = Lambda(
                                    upsampling2d_nearest_5d,
                                    arguments={
                                        'upsampling_factor_depth': upsampling_factor_depth,
                                        'upsampling_factor_height': upsampling_factor_height,
                                        'upsampling_factor_width': upsampling_factor_width
                                    }
                                )(inp)
                            else:
                                print(f'The Interpolate - {mode} is not yet implemented.')
                                sys.exit(-1)

                        else:
                            # Others
                            if mode == 'linear':
                                if output_edgetpu:
                                    # Dpeth (height x width)
                                    resized_list = []
                                    unstack_img_list = tf.unstack(inp, axis=1)
                                    for i in unstack_img_list:
                                        resized_list.append(
                                            tf.compat.v1.image.resize(
                                                i,
                                                [out_height, out_width],
                                                method='bilinear'
                                            )
                                        )
                                    stack_img_hw = tf.stack(resized_list, axis=1)
                                    # Width (depth x Height)
                                    resized_list = []
                                    unstack_img_list = tf.unstack(stack_img_hw, axis=3)
                                    for i in unstack_img_list:
                                        resized_list.append(
                                            tf.compat.v1.image.resize(
                                                i,
                                                [out_depth, out_height],
                                                method='bilinear'
                                            )
                                        )
                                    inp = tf.stack(resized_list, axis=3)

                                else:
                                    # Dpeth (height x width)
                                    resized_list = []
                                    unstack_img_list = tf.unstack(inp, axis=1)
                                    for i in unstack_img_list:
                                        resized_list.append(
                                            tf.image.resize(
                                                i,
                                                [out_height, out_width],
                                                method='bilinear'
                                            )
                                        )
                                    stack_img_hw = tf.stack(resized_list, axis=1)
                                    # Width (depth x Height)
                                    resized_list = []
                                    unstack_img_list = tf.unstack(stack_img_hw, axis=3)
                                    for i in unstack_img_list:
                                        resized_list.append(
                                            tf.image.resize(
                                                i,
                                                [out_depth, out_height],
                                                method='bilinear'
                                            )
                                        )
                                    inp = tf.stack(resized_list, axis=3)

                            elif mode == 'nearest':
                                if output_edgetpu:
                                    # Dpeth (height x width)
                                    resized_list = []
                                    unstack_img_list = tf.unstack(inp, axis=1)
                                    for i in unstack_img_list:
                                        resized_list.append(
                                            tf.compat.v1.image.resize(
                                                i,
                                                [out_height, out_width],
                                                method='nearest'
                                            )
                                        )
                                    stack_img_hw = tf.stack(resized_list, axis=1)
                                    # Width (depth x Height)
                                    resized_list = []
                                    unstack_img_list = tf.unstack(stack_img_hw, axis=3)
                                    for i in unstack_img_list:
                                        resized_list.append(
                                            tf.compat.v1.image.resize(
                                                i,
                                                [out_depth, out_height],
                                                method='nearest'
                                            )
                                        )
                                    inp = tf.stack(resized_list, axis=3)

                                else:
                                    # Dpeth (height x width)
                                    resized_list = []
                                    unstack_img_list = tf.unstack(inp, axis=1)
                                    for i in unstack_img_list:
                                        resized_list.append(
                                            tf.image.resize(
                                                i,
                                                [out_height, out_width],
                                                method='nearest'
                                            )
                                        )
                                    stack_img_hw = tf.stack(resized_list, axis=1)
                                    # Width (depth x Height)
                                    resized_list = []
                                    unstack_img_list = tf.unstack(stack_img_hw, axis=3)
                                    for i in unstack_img_list:
                                        resized_list.append(
                                            tf.image.resize(
                                                i,
                                                [out_depth, out_height],
                                                method='nearest'
                                            )
                                        )
                                    inp = tf.stack(resized_list, axis=3)

                            else:
                                print(f'The Interpolate - {mode} is not yet implemented.')
                                sys.exit(-1)

                    else:
                        if mode == 'linear':
                            # Dpeth (height x width)
                            resized_list = []
                            unstack_img_list = tf.unstack(inp, axis=1)
                            for i in unstack_img_list:
                                resized_list.append(
                                    tf.image.resize(
                                        i,
                                        [out_height, out_width],
                                        method='bilinear'
                                    )
                                )
                            stack_img_hw = tf.stack(resized_list, axis=1)
                            # Width (depth x Height)
                            resized_list = []
                            unstack_img_list = tf.unstack(stack_img_hw, axis=3)
                            for i in unstack_img_list:
                                resized_list.append(
                                    tf.image.resize(
                                        i,
                                        [out_depth, out_height],
                                        method='bilinear'
                                    )
                                )
                            inp = tf.stack(resized_list, axis=3)

                        elif mode == 'nearest':
                            # Dpeth (height x width)
                            resized_list = []
                            unstack_img_list = tf.unstack(inp, axis=1)
                            for i in unstack_img_list:
                                resized_list.append(
                                    tf.image.resize(
                                        i,
                                        [out_height, out_width],
                                        method='nearest'
                                    )
                                )
                            stack_img_hw = tf.stack(resized_list, axis=1)
                            # Width (depth x Height)
                            resized_list = []
                            unstack_img_list = tf.unstack(stack_img_hw, axis=3)
                            for i in unstack_img_list:
                                resized_list.append(
                                    tf.image.resize(
                                        i,
                                        [out_depth, out_height],
                                        method='nearest'
                                    )
                                )
                            inp = tf.stack(resized_list, axis=3)

                        else:
                            print(f'The Interpolate - {mode} is not yet implemented.')
                            sys.exit(-1)

                    if wr_config and layer_id in wr_config and format_version >= 2:
                        if wr_config[layer_id]['replace_mode'] == 'insert_before':
                            tf_layers_dict[layer_id] = inp

                        elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                            tf_layers_dict[layer_id] = extrapolation_of_layers(
                                wr_config[layer_id],
                                inp
                            )
                    else:
                        tf_layers_dict[layer_id] = inp

                else:
                    print(f'{Color.RED}ERROR:{Color.RESET} Interpolate is only supported in 4D or 5D. layer_id: {layer_id} input_shape: {input_shape}')
                    sys.exit(-1)

            ### ShapeOf
            elif layer.attrib['type'] == 'ShapeOf':
                try:
                    if wr_config and layer_id in wr_config and format_version >= 2:
                        if wr_config[layer_id]['replace_mode'] == 'insert_before':
                            inp = extrapolation_of_layers(
                                wr_config[layer_id],
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                            )
                            tf_layers_dict[layer_id] = tf.constant(
                                np.asarray(inp.type_spec.shape),
                                dtype=tf.int64
                            )

                        elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                            inp = tf.constant(
                                np.asarray(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].type_spec.shape),
                                dtype=tf.int64
                            )
                            tf_layers_dict[layer_id] = extrapolation_of_layers(
                                wr_config[layer_id],
                                inp
                            )

                    else:
                        tf_layers_dict[layer_id] = tf.constant(
                            np.asarray(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].type_spec.shape),
                            dtype=tf.int64
                        )

                except:
                    if wr_config and layer_id in wr_config and format_version >= 2:
                        if wr_config[layer_id]['replace_mode'] == 'insert_before':
                            inp = extrapolation_of_layers(
                                wr_config[layer_id],
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                            )
                            tf_layers_dict[layer_id] = tf.shape(
                                inp,
                                out_type=tf.int64
                            )

                        elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                            inp = tf.shape(
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                out_type=tf.int64
                            )
                            tf_layers_dict[layer_id] = extrapolation_of_layers(
                                wr_config[layer_id],
                                inp
                            )

                    else:
                        tf_layers_dict[layer_id] = tf.shape(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            out_type=tf.int64
                        )

            ### Convert
            elif layer.attrib['type'] == 'Convert':
                # vino:    u8,    u16,    u32,    u64,   i8,   i16,   i32,   i64,     f16,     f32,              bf16, boolean
                # tf  : uint8, uint16, uint32, uint64, int8, int16, int32, int64, float16, float32, float64, bfloat16
                destination_type = data.attrib['destination_type']

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.cast(
                            inp,
                            cast_type_ov_tf[destination_type]
                        )

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.cast(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            cast_type_ov_tf[destination_type]
                        )
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )

                else:
                    tf_layers_dict[layer_id] = tf.cast(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        cast_type_ov_tf[destination_type]
                    )

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
                if ellipsis_mask:
                    ellipsis_mask = np.asarray([int(val) for val in ellipsis_mask.split(',')])
                else:
                    ellipsis_mask = np.asarray([0 for i in range(len(begin_mask))])
                if new_axis_mask:
                    new_axis_mask    = np.asarray([int(val) for val in new_axis_mask.split(',')])
                else:
                    new_axis_mask = np.asarray([0 for i in range(len(begin_mask))])
                if shrink_axis_mask:
                    shrink_axis_mask = np.asarray([int(val) for val in shrink_axis_mask.split(',')])
                else:
                    shrink_axis_mask = np.asarray([0 for i in range(len(begin_mask))])

                if type(begin_mask) == np.ndarray and len(begin_mask) == 4:
                    begin_mask[0], begin_mask[1], begin_mask[2], begin_mask[3] = \
                        begin_mask[0], begin_mask[2], begin_mask[3], begin_mask[1]
                if np.sum(begin_mask) == len(begin_mask):
                    begin_mask = -1
                else:
                    begin_mask = np.argmin(begin_mask)

                if type(end_mask) == np.ndarray and len(end_mask) == 4:
                    end_mask[0], end_mask[1], end_mask[2], end_mask[3] = \
                        end_mask[0], end_mask[2], end_mask[3], end_mask[1]
                if np.sum(end_mask) == len(end_mask):
                    end_mask = -1
                else:
                    end_mask = np.argmin(end_mask)

                if type(ellipsis_mask) == np.ndarray and len(ellipsis_mask) == 4:
                    ellipsis_mask[0], ellipsis_mask[1], ellipsis_mask[2], ellipsis_mask[3] = \
                        ellipsis_mask[0], ellipsis_mask[2], ellipsis_mask[3], ellipsis_mask[1]
                ellipsis_mask = np.argmin(ellipsis_mask)

                if type(new_axis_mask) == np.ndarray and len(new_axis_mask) == 4:
                    new_axis_mask[0], new_axis_mask[1], new_axis_mask[2], new_axis_mask[3] = \
                        new_axis_mask[0], new_axis_mask[2], new_axis_mask[3], new_axis_mask[1]
                new_axis_mask = np.argmin(new_axis_mask)


                if type(shrink_axis_mask) == np.ndarray and len(shrink_axis_mask) == 4:
                    shrink_axis_mask[0], shrink_axis_mask[1], shrink_axis_mask[2], shrink_axis_mask[3] = \
                        shrink_axis_mask[0], shrink_axis_mask[2], shrink_axis_mask[3], shrink_axis_mask[1]
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

                # Replacement
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['type'] == 'StridedSlice' and wr_config[layer_id]['replace_mode'] == 'change_attributes':
                        begin_mask = int(wr_config[layer_id]['values'][0])
                        end_mask = int(wr_config[layer_id]['values'][1])
                        ellipsis_mask = int(wr_config[layer_id]['values'][2])
                        new_axis_mask = int(wr_config[layer_id]['values'][3])
                        shrink_axis_mask = int(wr_config[layer_id]['values'][4])

                    elif wr_config[layer_id]['type'] == 'StridedSlice' and wr_config[layer_id]['replace_mode'] == 'replace':
                        begin = wr_config[layer_id]['values'][0]
                        end = wr_config[layer_id]['values'][1]
                        strides = wr_config[layer_id]['values'][2]
                        begin_mask = int(wr_config[layer_id]['values'][3])
                        end_mask = int(wr_config[layer_id]['values'][4])
                        ellipsis_mask = int(wr_config[layer_id]['values'][5])
                        new_axis_mask = int(wr_config[layer_id]['values'][6])
                        shrink_axis_mask = int(wr_config[layer_id]['values'][7])

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.strided_slice(
                            inp,
                            begin=begin,
                            end=end,
                            strides=strides,
                            begin_mask=begin_mask,
                            end_mask=end_mask,
                            ellipsis_mask=ellipsis_mask,
                            new_axis_mask=new_axis_mask,
                            shrink_axis_mask=shrink_axis_mask
                        )

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.strided_slice(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            begin=begin,
                            end=end,
                            strides=strides,
                            begin_mask=begin_mask,
                            end_mask=end_mask,
                            ellipsis_mask=ellipsis_mask,
                            new_axis_mask=new_axis_mask,
                            shrink_axis_mask=shrink_axis_mask
                        )
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                    else:
                        tf_layers_dict[layer_id] = tf.strided_slice(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            begin=begin,
                            end=end,
                            strides=strides,
                            begin_mask=begin_mask,
                            end_mask=end_mask,
                            ellipsis_mask=ellipsis_mask,
                            new_axis_mask=new_axis_mask,
                            shrink_axis_mask=shrink_axis_mask
                        )

                else:
                    tf_layers_dict[layer_id] = tf.strided_slice(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        begin=begin,
                        end=end,
                        strides=strides,
                        begin_mask=begin_mask,
                        end_mask=end_mask,
                        ellipsis_mask=ellipsis_mask,
                        new_axis_mask=new_axis_mask,
                        shrink_axis_mask=shrink_axis_mask
                    )

            ### Pad
            elif layer.attrib['type'] == 'Pad':
                pad_mode = pad_type_ov_tf[data.attrib['pad_mode']]
                pads_begin = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)] # [0,0,1,1]
                pads_end   = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)] # [0,0,1,1]

                if pad_mode != 'CONSTANT':
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

                else:
                    paddings = [pads_begin, pads_end]

                pad_value  = np.float32(0.0)
                if 'pad_value' in data.attrib:
                    pad_value = np.float32(data.attrib['pad_value'])

                inp = None
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                else:
                    inp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]

                try:
                    inp = tf.pad(
                        inp,
                        paddings,
                        mode=pad_mode,
                        constant_values=pad_value
                    )
                except:
                    # workaround
                    # pads_begin: [0,0,0,1] -> [n,c,h,w]
                    # pads_end  : [0,0,2,0] -> [n,c,h,w]
                    # paddins = [[0,2], [1,0]] #[[top, bottom], [left, right]]
                    if len(inp.shape) == 4:
                        paddings = [
                            [pads_begin[2], pads_end[2]],
                            [pads_begin[3], pads_end[3]]
                        ]
                        inp = tf.keras.layers.ZeroPadding2D(
                            padding=paddings
                        )(inp)
                    else:
                        inp = tf.identity(
                            inp
                        )

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        tf_layers_dict[layer_id] = inp

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### TopK
            elif layer.attrib['type'] == 'TopK':
                # axis = int(data.attrib['axis'])
                # index_element_type = data.attrib['index_element_type']
                # mode = data.attrib['mode']
                # sort = data.attrib['sort']
                layer_id_values  = layer_id_port_dict[layer_id]['layer_id:port'][0]
                layer_id_indices = layer_id_port_dict[layer_id]['layer_id:port'][1]

                def _nnapi_scalar(value, dtype):
                    return tf.constant(value, dtype=dtype, shape=(1,))

                def _alternative_argmax(
                    input_tensor,
                    axis=-1,
                    output_type = tf.dtypes.float32,
                    name = None,
                    keepdims = False,
                    epsilon = None,
                    replace_argmax_indices_to_float32 = False
                ):
                    safe_axis = axis
                    if safe_axis < 0:
                        safe_axis = len(input_tensor.shape) + safe_axis
                    reduction_size = input_tensor.shape[axis]
                    axis_max = tf.math.reduce_max(input_tensor, axis=axis, keepdims=True)
                    zero_if_max = tf.subtract(axis_max, input_tensor)
                    eps = epsilon if epsilon else 1e-6
                    if input_tensor.dtype.is_floating:
                        zero_if_max_else_eps = tf.math.minimum(_nnapi_scalar(eps, input_tensor.dtype), zero_if_max)
                        zero_if_max_else_one = zero_if_max_else_eps * _nnapi_scalar(1 / eps, input_tensor.dtype)
                    elif input_tensor.dtype.is_integer:
                        zero_if_max_else_one = tf.math.minimum(_nnapi_scalar(1, input_tensor.dtype), zero_if_max)
                    else:
                        raise ValueError('Please specify epsilon for unknown input data type')

                    zero_if_max_else_one = tf.cast(zero_if_max_else_one, dtype=output_type)
                    zero_if_max_else_one = zero_if_max_else_one
                    one_if_max_else_zero = tf.math.subtract(_nnapi_scalar(1, output_type), zero_if_max_else_one)
                    rev_index = tf.range(reduction_size, 0, -1, dtype=output_type)
                    for index in range(safe_axis + 1, len(input_tensor.shape)):
                        rev_index = tf.expand_dims(rev_index, axis=index - safe_axis)
                    rev_index = rev_index
                    rev_index_if_max_else_zero = tf.math.multiply(one_if_max_else_zero, rev_index)
                    reverse_argmax = tf.math.reduce_max(rev_index_if_max_else_zero, axis=axis, keepdims=keepdims, name=name)
                    if not replace_argmax_indices_to_float32:
                        return tf.cast(tf.math.subtract(_nnapi_scalar(reduction_size, output_type), reverse_argmax, name=name), dtype=tf.int32)
                    else:
                        return tf.math.subtract(_nnapi_scalar(reduction_size, output_type), reverse_argmax, name=name)

                k = int(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])
                try:
                    if wr_config and layer_id in wr_config and format_version >= 2:
                        if wr_config[layer_id]['replace_mode'] == 'insert_before':
                            inp = extrapolation_of_layers(
                                wr_config[layer_id],
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                            )
                            if k != 1:
                                tf_layers_dict[layer_id_values], tf_layers_dict[layer_id_indices] = \
                                    tf.math.top_k(
                                        inp,
                                        k=k,
                                        sorted=True
                                    )
                            else:
                                if not replace_argmax:
                                    tf_layers_dict[layer_id_values], tf_layers_dict[layer_id_indices] = \
                                        tf.math.top_k(
                                            inp,
                                            k=k,
                                            sorted=True
                                        )
                                else:
                                    tf_layers_dict[layer_id_indices] = _alternative_argmax(
                                        input_tensor=inp,
                                        axis=-1,
                                        output_type=tf.float32,
                                        keepdims=True,
                                        replace_argmax_indices_to_float32=replace_argmax_indices_to_float32
                                    )
                                    tf_layers_dict[layer_id_values] = tf.reduce_max(
                                        inp,
                                        axis=-1
                                    )

                        elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                            print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                            sys.exit(-1)

                    else:
                        if k != 1:
                            tf_layers_dict[layer_id_values], tf_layers_dict[layer_id_indices] = \
                                tf.math.top_k(
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                    k=k,
                                    sorted=True
                                )
                        else:
                            if not replace_argmax:
                                tf_layers_dict[layer_id_values], tf_layers_dict[layer_id_indices] = \
                                    tf.math.top_k(
                                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                        k=k,
                                        sorted=True
                                    )
                            else:
                                tf_layers_dict[layer_id_indices] = _alternative_argmax(
                                    input_tensor=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                    axis=-1,
                                    output_type=tf.float32,
                                    keepdims=True,
                                    replace_argmax_indices_to_float32=replace_argmax_indices_to_float32
                                )
                                tf_layers_dict[layer_id_values] = tf.reduce_max(
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                    axis=-1
                                )

                except:
                    k = int(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)][0])
                    if wr_config and layer_id in wr_config and format_version >= 2:
                        if wr_config[layer_id]['replace_mode'] == 'insert_before':
                            inp = extrapolation_of_layers(
                                wr_config[layer_id],
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                            )
                            if k != 1:
                                tf_layers_dict[layer_id_values], tf_layers_dict[layer_id_indices] = \
                                    tf.math.top_k(
                                        inp,
                                        k=k,
                                        sorted=True
                                    )
                            else:
                                if not replace_argmax:
                                    tf_layers_dict[layer_id_values], tf_layers_dict[layer_id_indices] = \
                                        tf.math.top_k(
                                            inp,
                                            k=k,
                                            sorted=True
                                        )
                                else:
                                    tf_layers_dict[layer_id_indices] = _alternative_argmax(
                                        input_tensor=inp,
                                        axis=-1,
                                        output_type=tf.float32,
                                        keepdims=True,
                                        replace_argmax_indices_to_float32=replace_argmax_indices_to_float32
                                    )
                                    tf_layers_dict[layer_id_values] = tf.reduce_max(
                                        inp,
                                        axis=-1
                                    )

                        elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                            print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                            sys.exit(-1)

                    else:
                        if k != 1:
                            tf_layers_dict[layer_id_values], tf_layers_dict[layer_id_indices] = \
                                tf.math.top_k(
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                    k=k,
                                    sorted=True
                                )
                        else:
                            if not replace_argmax:
                                tf_layers_dict[layer_id_values], tf_layers_dict[layer_id_indices] = \
                                    tf.math.top_k(
                                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                        k=k,
                                        sorted=True
                                    )
                            else:
                                tf_layers_dict[layer_id_indices] = _alternative_argmax(
                                    input_tensor=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                    axis=-1,
                                    output_type=tf.float32,
                                    keepdims=True,
                                    replace_argmax_indices_to_float32=replace_argmax_indices_to_float32
                                )
                                tf_layers_dict[layer_id_values] = tf.reduce_max(
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                    axis=-1
                                )

            ### Transpose
            elif layer.attrib['type'] == 'Transpose':
                input_shape_len = len(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape)
                temp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                perm = []
                if input_shape_len == 4:
                    if type(temp) == np.ndarray:
                        if np.all(temp == [0,3,1,2]):
                            for idx, dim in enumerate(temp):
                                perm.append(idx)
                        elif np.all(temp == [1,0,2,3]):
                            perm = [3,1,2,0]
                        else:
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
                    if np.all(temp == [0,2,1,3,4]):
                        perm.append(0)
                        perm.append(1)
                        perm.append(2)
                        perm.append(4)
                        perm.append(3)
                    else:
                        # TODO
                        for idx, dim in enumerate(temp):
                            perm.append(dim)
                else:
                    for idx, dim in enumerate(temp):
                        perm.append(dim)

                # If the Transpose shape has been changed by extrapolation, the extrapolated perm is given priority.
                try:
                    before_layer_id = str(int(layer_id) - 1)
                    if wr_config and before_layer_id in wr_config and format_version >= 2:
                        if wr_config[before_layer_id]['type'] == 'Const':
                            perm = wr_config[before_layer_id]['values']
                except:
                    pass

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.transpose(
                            inp,
                            perm=perm
                        )
                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.transpose(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            perm=perm
                        )
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.transpose(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        perm=perm
                    )

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

                # If the Squeeze axis has been changed by extrapolation, the extrapolated axis is given priority.
                try:
                    before_layer_id = str(int(layer_id) - 1)
                    if wr_config and before_layer_id in wr_config and format_version >= 2:
                        if wr_config[before_layer_id]['type'] == 'Const':
                            axis = wr_config[before_layer_id]['values']
                except:
                    pass

                try:
                    if wr_config and layer_id in wr_config and format_version >= 2:
                        if wr_config[layer_id]['replace_mode'] == 'insert_before':
                            inp = extrapolation_of_layers(
                                wr_config[layer_id],
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                            )
                            tf_layers_dict[layer_id] = tf.squeeze(
                                inp,
                                axis=axis
                            )
                        elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                            inp = tf.squeeze(
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                axis=axis
                            )
                            tf_layers_dict[layer_id] = extrapolation_of_layers(
                                wr_config[layer_id],
                                inp
                            )
                    else:
                        tf_layers_dict[layer_id] = tf.squeeze(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            axis=axis
                        )

                except:
                    if wr_config and layer_id in wr_config and format_version >= 2:
                        if wr_config[layer_id]['replace_mode'] == 'insert_before':
                            inp = extrapolation_of_layers(
                                wr_config[layer_id],
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                            )
                            tf_layers_dict[layer_id] = tf.squeeze(
                                inp,
                                axis=-1
                            )

                        elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                            inp = tf.squeeze(
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                axis=-1
                            )
                            tf_layers_dict[layer_id] = extrapolation_of_layers(
                                wr_config[layer_id],
                                inp
                            )
                    else:
                        tf_layers_dict[layer_id] = tf.squeeze(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            axis=-1
                        )

            ### Gather
            elif layer.attrib['type'] == 'Gather':
                axis = int(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)])
                temp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                input_shape = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape[0]
                output_shape = [int(v.text) for v in layer.find("output").find("port")]

                batch_dims = 0
                if not data is None and 'batch_dims' in data.attrib:
                    batch_dims = int(data.attrib['batch_dims'])
                    if batch_dims == 0:
                        batch_dims = None
                indices = []
                if type(temp) == np.ndarray:
                    if temp.ndim == 1:
                        for idx, dim in enumerate(temp):
                            indices.append(dim)
                    else:
                        indices = temp
                else:
                    # TODO
                    if len(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape) < len(temp.shape):
                        for idx, dim in enumerate(temp):
                            indices.append(dim)
                    elif len(temp.shape) == 1:
                        indices = temp
                    else:
                        shape = tf.shape(temp)
                        for idx, dim in enumerate(shape):
                            if idx == 0:
                                indices.append(0)
                            elif idx == input_shape - 1:
                                indices.append(1)
                            else:
                                indices.append(dim + 1)

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        if isinstance(inp, tf.Tensor):
                            tf_layers_dict[layer_id] = tf.gather(
                                inp,
                                indices,
                                axis=axis,
                                batch_dims=batch_dims
                            )
                        else:
                            try:
                                if indices == [0] and axis == 0:
                                    try:
                                        tmp = tf.squeeze(
                                            inp,
                                            axis=axis
                                        )
                                        if tmp.type_spec.shape == []:
                                            tf_layers_dict[layer_id] = tf.expand_dims(tmp, axis=0)
                                    except:
                                        tf_layers_dict[layer_id] = tf.gather(
                                            inp,
                                            indices,
                                            axis=axis,
                                            batch_dims=batch_dims
                                        )
                                else:
                                    tf_layers_dict[layer_id] = tf.gather(
                                        inp,
                                        indices,
                                        axis=axis,
                                        batch_dims=batch_dims
                                    )
                            except:
                                tf_layers_dict[layer_id] = tf.gather(
                                    inp,
                                    indices,
                                    axis=axis,
                                    batch_dims=batch_dims
                                )

                            if batch_dims is None and axis == 0 and tf_layers_dict[layer_id].shape[0] == 1:
                                tf_layers_dict[layer_id] = tf_layers_dict[layer_id][0]
                            elif batch_dims is None and (axis == -1 or axis == (len(tf_layers_dict[layer_id].shape) - 1)) and tf_layers_dict[layer_id].shape[-1] == 1:
                                tf_layers_dict[layer_id] = tf.squeeze(tf_layers_dict[layer_id], axis=axis)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        if isinstance(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf.Tensor):
                            inp = tf.gather(
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                indices,
                                axis=axis,
                                batch_dims=batch_dims
                            )
                        else:
                            try:
                                if indices == [0] and axis == 0:
                                    try:
                                        inp = tf.squeeze(
                                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                            axis=axis
                                        )
                                        if tf_layers_dict[layer_id].type_spec.shape == []:
                                            inp = tf.expand_dims(tf_layers_dict[layer_id], axis=0)
                                    except:
                                        inp = tf.gather(
                                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                            indices,
                                            axis=axis,
                                            batch_dims=batch_dims
                                        )
                                else:
                                    inp = tf.gather(
                                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                        indices,
                                        axis=axis,
                                        batch_dims=batch_dims
                                    )
                            except:
                                inp = tf.gather(
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                    indices,
                                    axis=axis,
                                    batch_dims=batch_dims
                                )

                            if batch_dims is None and axis == 0 and inp.shape[0] == 1:
                                inp = inp[0]
                            elif batch_dims is None and (axis == -1 or axis == (len(inp.shape) - 1)) and inp.shape[-1] == 1:
                                inp = tf.squeeze(inp, axis=axis)

                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )

                else:
                    if isinstance(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf.Tensor):
                        tf_layers_dict[layer_id] = tf.gather(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            indices,
                            axis=axis,
                            batch_dims=batch_dims
                        )
                    else:
                        try:
                            if indices == [0] and axis == 0:
                                try:
                                    tf_layers_dict[layer_id] = tf.squeeze(
                                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                        axis=axis
                                    )
                                    if tf_layers_dict[layer_id].type_spec.shape == []:
                                        tf_layers_dict[layer_id] = tf.expand_dims(tf_layers_dict[layer_id], axis=0)
                                except:
                                    tf_layers_dict[layer_id] = tf.gather(
                                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                        indices,
                                        axis=axis,
                                        batch_dims=batch_dims
                                    )
                            else:
                                tf_layers_dict[layer_id] = tf.gather(
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                    indices,
                                    axis=axis,
                                    batch_dims=batch_dims
                                )

                                if batch_dims is None and axis == 0 and tf_layers_dict[layer_id].shape[0] == 1:
                                    tf_layers_dict[layer_id] = tf_layers_dict[layer_id][0]
                                elif batch_dims is None and (axis == -1 or axis == (len(tf_layers_dict[layer_id].shape) - 1)) and tf_layers_dict[layer_id].shape[-1] == 1:
                                    tf_layers_dict[layer_id] = tf.squeeze(tf_layers_dict[layer_id], axis=axis)
                        except:
                            tf_layers_dict[layer_id] = tf.gather(
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                indices,
                                axis=axis,
                                batch_dims=batch_dims
                            )

                            if batch_dims is None and axis == 0 and tf_layers_dict[layer_id].shape[0] == 1:
                                tf_layers_dict[layer_id] = tf_layers_dict[layer_id][0]
                            elif batch_dims is None and (axis == -1 or axis == (len(tf_layers_dict[layer_id].shape) - 1)) and tf_layers_dict[layer_id].shape[-1] == 1:
                                tf_layers_dict[layer_id] = tf.squeeze(tf_layers_dict[layer_id], axis=axis)

            ### GatherND
            elif layer.attrib['type'] == 'GatherND':
                batch_dims = int(data.attrib['batch_dims'])
                params = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                indices_tmp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                if indices_tmp.dtype is tf.float32 or indices_tmp.dtype is tf.float64:
                    indices = tf.cast(indices_tmp, tf.int32)
                else:
                    indices = indices_tmp

                def barracuda_gather_nd(params, indices):
                    idx_shape = indices.shape
                    params_shape = params.shape
                    idx_dims = idx_shape[-1]
                    gather_shape = params_shape[idx_dims:]
                    params_flat = tf.reshape(params, tf.concat([[-1], gather_shape], axis=0))
                    axis_step = tf.math.cumprod(params_shape[:idx_dims], exclusive=True, reverse=True)
                    mul = tf.math.multiply(indices, axis_step)
                    indices_flat = tf.reduce_sum(mul, axis=-1)
                    result_flat = tf.gather(params_flat, indices_flat)
                    return tf.reshape(result_flat, tf.concat([idx_shape[:-1], gather_shape], axis=0))

                if optimizing_barracuda and batch_dims > 0:
                    print(f'{Color.RED}ERROR:{Color.RESET} When optimize_barracuda = True, batch_dims > 0 is not supported. layer_id: {layer_id}')
                    sys.exit(-1)

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        if not optimizing_barracuda:
                            tf_layers_dict[layer_id] = tf.gather_nd(inp, indices, batch_dims=batch_dims)
                        else:
                            tf_layers_dict[layer_id] = barracuda_gather_nd(inp, indices)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = None
                        if not optimizing_barracuda:
                            inp = tf.gather_nd(params, indices, batch_dims=batch_dims)
                        else:
                            inp = barracuda_gather_nd(params, indices)

                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    if not optimizing_barracuda:
                        tf_layers_dict[layer_id] = tf.gather_nd(params, indices, batch_dims=batch_dims)
                    else:
                        tf_layers_dict[layer_id] = barracuda_gather_nd(params, indices)

            ### ReduceMean, ReduceMax, ReduceMin, ReduceSum, ReduceProd, ReduceL2 - TODO
            elif layer.attrib['type'] == 'ReduceMean' or layer.attrib['type'] == 'ReduceMax' or layer.attrib['type'] == 'ReduceMin' or \
                layer.attrib['type'] == 'ReduceSum' or layer.attrib['type'] == 'ReduceProd' or \
                layer.attrib['type'] == 'ReduceL1' or layer.attrib['type'] == 'ReduceL2':
                keep_dims = True if (data.attrib['keep_dims'] == "True" or data.attrib['keep_dims'] == "true") else False
                axis = None
                if type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]) == np.ndarray and len(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]) == 1:
                    axis = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].astype(np.int32)
                    if axis == 1:
                        axis = -1
                    elif axis >= 2:
                        axis -= 1
                elif type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]) != np.ndarray and len(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].shape) == 1:
                    try:
                        if (tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].numpy() == [1, 2]).all():
                            if type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]).dtype != tf.int32:
                                axis = tf.cast(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)], tf.int32)
                            else:
                                axis = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                        else:
                            if type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]).dtype != tf.int32:
                                axis = tf.cast(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)] - 1, tf.int32)
                            else:
                                axis = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)] - 1
                    except:
                        if type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]).dtype != tf.int32:
                            axis = tf.cast(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)] - 1, tf.int32)
                        else:
                            axis = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)] - 1
                else:
                    for idx, part_axis in enumerate(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]):
                        if part_axis == 1:
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)][idx] = -1
                        elif part_axis >= 2:
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)][idx] -= 1
                    if type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]).dtype != tf.int32:
                        axis = tf.cast(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)], tf.int32)
                    else:
                        axis = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]

                inp = None
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                else:
                    inp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]

                if layer.attrib['type'] == 'ReduceMean':
                    inp = tf.math.reduce_mean(
                        inp,
                        axis=axis,
                        keepdims=keep_dims
                    )
                elif layer.attrib['type'] == 'ReduceMax':
                    inp = tf.math.reduce_max(
                        inp,
                        axis=axis,
                        keepdims=keep_dims
                    )
                elif layer.attrib['type'] == 'ReduceMin':
                    inp = tf.math.reduce_min(
                        inp,
                        axis=axis,
                        keepdims=keep_dims
                    )
                elif layer.attrib['type'] == 'ReduceSum':
                    inp = tf.math.reduce_sum(
                        inp,
                        axis=axis,
                        keepdims=keep_dims
                    )
                elif layer.attrib['type'] == 'ReduceProd':
                    inp = tf.math.reduce_prod(
                        inp,
                        axis=axis,
                        keepdims=keep_dims
                    )
                elif layer.attrib['type'] == 'ReduceL1':
                    reduceL1_abs = tf.math.abs(
                        inp
                    )
                    inp = tf.math.reduce_sum(
                        reduceL1_abs,
                        axis=axis,
                        keepdims=keep_dims
                    )
                elif layer.attrib['type'] == 'ReduceL2':
                    reduceL2_square = tf.math.square(
                        inp
                    )
                    reduceL2_sum = tf.math.reduce_sum(
                        reduceL2_square,
                        axis=axis,
                        keepdims=keep_dims
                    )
                    inp = tf.math.sqrt(reduceL2_sum)

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        tf_layers_dict[layer_id] = inp

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

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
                        transpose_a = True if (data.attrib['transpose_a'] == 'True' or data.attrib['transpose_a'] == 'true') else False
                if not data is None and 'transpose_b' in data.attrib:
                    try:
                        transpose_b = True if int(data.attrib['transpose_b']) == 1 else False
                    except:
                        transpose_b = True if (data.attrib['transpose_b'] == 'True'or data.attrib['transpose_b'] == 'true') else False

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        try:
                            tf_layers_dict[layer_id] = tf.linalg.matmul(
                                inp,
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                                transpose_a,
                                transpose_b
                            )
                        except Exception as e:
                            sub_inp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                            if len(inp.shape) == 2 and len(sub_inp.shape) == 2:
                                inp1 = tf.transpose(inp, perm=[1,0])
                                inp2 = tf.transpose(sub_inp, perm=[1,0])
                                tf_layers_dict[layer_id] = tf.linalg.matmul(
                                    inp1,
                                    inp2,
                                    transpose_a,
                                    transpose_b
                                )
                            elif len(inp.shape) == 3 and len(sub_inp.shape) == 3:
                                inp1 = tf.transpose(inp, perm=[0,2,1])
                                inp2 = tf.transpose(sub_inp, perm=[0,2,1])
                                tf_layers_dict[layer_id] = tf.linalg.matmul(
                                    inp1,
                                    inp2,
                                    transpose_a,
                                    transpose_b
                                )
                            elif len(inp.shape) == 3 and len(sub_inp.shape) == 2:
                                inp1 = tf.transpose(inp, perm=[0,2,1])
                                tf_layers_dict[layer_id] = tf.linalg.matmul(
                                    inp1,
                                    sub_inp,
                                    transpose_a,
                                    transpose_b
                                )
                            else:
                                raise e

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        try:
                            inp = tf.linalg.matmul(
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                                transpose_a,
                                transpose_b
                            )
                        except Exception as e:
                            main_inp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                            sub_inp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                            if len(main_inp.shape) == 2 and len(sub_inp.shape) == 2:
                                inp1 = tf.transpose(main_inp, perm=[1,0])
                                inp2 = tf.transpose(sub_inp, perm=[1,0])
                                inp = tf.linalg.matmul(
                                    inp1,
                                    inp2,
                                    transpose_a,
                                    transpose_b
                                )
                            elif len(main_inp.shape) == 3 and len(sub_inp.shape) == 3:
                                inp1 = tf.transpose(main_inp, perm=[0,2,1])
                                inp2 = tf.transpose(sub_inp, perm=[0,2,1])
                                inp = tf.linalg.matmul(
                                    inp1,
                                    inp2,
                                    transpose_a,
                                    transpose_b
                                )
                            elif len(main_inp.shape) == 3 and len(sub_inp.shape) == 2:
                                inp1 = tf.transpose(main_inp, perm=[0,2,1])
                                inp = tf.linalg.matmul(
                                    inp1,
                                    sub_inp,
                                    transpose_a,
                                    transpose_b
                                )
                            else:
                                raise e

                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )

                else:
                    try:
                        tf_layers_dict[layer_id] = tf.linalg.matmul(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                            transpose_a,
                            transpose_b
                        )
                    except Exception as e:
                        main_inp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        sub_inp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                        if len(main_inp.shape) == 2 and len(sub_inp.shape) == 2:
                            main_inp = tf.transpose(main_inp, perm=[1,0])
                            sub_inp = tf.transpose(sub_inp, perm=[1,0])
                            tf_layers_dict[layer_id] = tf.linalg.matmul(
                                main_inp,
                                sub_inp,
                                transpose_a,
                                transpose_b
                            )
                        elif len(main_inp.shape) == 3 and len(sub_inp.shape) == 3:
                            main_inp = tf.transpose(main_inp, perm=[0,2,1])
                            sub_inp = tf.transpose(sub_inp, perm=[0,2,1])
                            tf_layers_dict[layer_id] = tf.linalg.matmul(
                                main_inp,
                                sub_inp,
                                transpose_a,
                                transpose_b
                            )
                        elif len(main_inp.shape) == 3 and len(sub_inp.shape) == 2:
                            main_inp = tf.transpose(main_inp, perm=[0,2,1])
                            tf_layers_dict[layer_id] = tf.linalg.matmul(
                                main_inp,
                                sub_inp,
                                transpose_a,
                                transpose_b
                            )
                        else:
                            raise e

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
                        shape = op2

                    elif op_len2 == 1 and op2.shape[0] == 6:
                        # YoloV4
                        shape = op2

                elif op_type1 != np.ndarray and op_type2 == np.ndarray:
                    # op and const
                    if op_len2 == 4:
                        op2 = op2.transpose(0,2,3,1)
                        shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]
                    elif op_len2 > 4:
                        shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]
                    elif op_len2 == 1 and op2.shape[0] == 1:
                        shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]
                    elif op_len2 == 1 and op2.shape[0] == 2:
                        shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]
                    elif op_len2 == 1 and op2.shape[0] == 3:
                        shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]
                        if op_len1 > len(op2):
                            if shape[2] == -1:
                                shape[0], shape[1], shape[2] = shape[0], shape[2], shape[1]

                    elif op_len2 == 1 and op2.shape[0] == 4:
                        one_count_1 = 0
                        for idx in op1.shape:
                            if idx == 1:
                                one_count_1 += 1
                        one_count_2 = 0
                        for idx in op2:
                            if idx == 1:
                                one_count_2 += 1
                        if one_count_1 != one_count_2 and one_count_2 == 3 and op2[3] != 1:
                            shape_tmp = []
                            shape_tmp.append(op2[0])
                            shape_tmp.append(op2[1])
                            shape_tmp.append(op2[2])
                            shape_tmp.append(op2[3])
                            shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(shape_tmp)]
                        else:
                            shape_tmp = []
                            shape_tmp.append(op2[0])
                            shape_tmp.append(op2[2])
                            shape_tmp.append(op2[3])
                            shape_tmp.append(op2[1])
                            shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(shape_tmp)]
                    elif op_len2 == 1 and op2.shape[0] == 5:
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
                        for i in range(op2.shape[0]):
                            shape.append(op2[i])

                elif op_type1 == np.ndarray and op_type2 != np.ndarray:
                    # const and op
                    if op_len1 == 4:
                        op1 = op1.transpose(0,2,3,1)
                        shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]
                    elif op_len1 > 4:
                        shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]
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

                # If the Reshape shape has been changed by extrapolation, the extrapolated shape is given priority.
                try:
                    before_layer_id = str(int(layer_id) - 1)
                    if wr_config and before_layer_id in wr_config and format_version >= 2:
                        if wr_config[before_layer_id]['type'] == 'Const':
                            shape = wr_config[before_layer_id]['values']
                except:
                    pass

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            op1
                        )
                        tf_layers_dict[layer_id] = tf.reshape(inp, shape=shape)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.reshape(op1, shape=shape)
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.reshape(op1, shape)

            ### Range - TODO
            elif layer.attrib['type'] == 'Range':
                dtype = cast_type_ov_tf[data.attrib['output_type']]
                start = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)][0]
                limit = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                try:
                    if start == 2 and 'Squeeze' in limit.name and type(limit.type_spec) == tf.TensorSpec and limit.type_spec.dtype == tf.int64:
                        start = 1
                        limit = tf.constant(3)
                except:
                    if start == 2 and limit.numpy() == 4:
                        start = 1
                        limit = tf.constant(3)

                inp = tf.range(
                    start,
                    limit,
                    delta=int(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]),
                    dtype=dtype
                )

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### Exp
            elif layer.attrib['type'] == 'Exp':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.math.exp(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.math.exp(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.math.exp(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Abs
            elif layer.attrib['type'] == 'Abs':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.math.abs(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.math.abs(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.math.abs(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### SoftMax
            elif layer.attrib['type'] == 'SoftMax':
                axis = int(data.attrib['axis'])
                if axis == 1 and len(np.asarray(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape)) == 4:
                    axis = -1

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['type'] == 'SoftMax' and wr_config[layer_id]['replace_mode'] == 'change_axis':
                        axis = int(wr_config[layer_id]['values'])

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.nn.softmax(
                            inp,
                            axis=axis
                        )

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.nn.softmax(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            axis=axis
                        )
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                    else:
                        tf_layers_dict[layer_id] = tf.nn.softmax(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            axis=axis
                        )
                else:
                    tf_layers_dict[layer_id] = tf.nn.softmax(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        axis=axis
                    )

            ### Negative
            elif layer.attrib['type'] == 'Negative':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        if not output_edgetpu:
                            tf_layers_dict[layer_id] = tf.math.negative(inp)
                        else:
                            tf_layers_dict[layer_id] = tf.math.multiply(inp, -1.0)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        if not output_edgetpu:
                            inp = tf.math.negative(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        else:
                            inp = tf.math.multiply(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], -1.0)
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    if not output_edgetpu:
                        tf_layers_dict[layer_id] = tf.math.negative(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                    else:
                        tf_layers_dict[layer_id] = tf.math.multiply(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], -1.0)

            ### Maximum
            elif layer.attrib['type'] == 'Maximum':
                # No broadcast
                tf_layers_dict[layer_id] = tf.math.maximum(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )
                if wr_config and layer_id in wr_config and format_version >= 2:
                    print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to "Maximum" is not supported. layer_id: {layer_id}')
                    sys.exit(-1)

            ### Minimum
            elif layer.attrib['type'] == 'Minimum':
                # No broadcast
                tf_layers_dict[layer_id] = tf.math.minimum(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )
                if wr_config and layer_id in wr_config and format_version >= 2:
                    print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to "Minimum" is not supported. layer_id: {layer_id}')
                    sys.exit(-1)

            ### Acos
            elif layer.attrib['type'] == 'Acos':

                # https://zenn.dev/pinto0309/articles/8f6df1d2304395
                def pseudo_acos(x, output_myriad):
                    x_abs = None
                    if output_myriad:
                        x_abs = tf.math.sqrt(tf.math.square(x)) # OAK-D (MyriadX) is not compatible with ABS.
                    else:
                        x_abs = tf.abs(x)
                    neg = tf.math.divide(tf.math.multiply(tf.minimum(x, 0), -1), x_abs)
                    x = x_abs
                    y = tf.constant(-0.0187293)
                    y = tf.math.multiply(y, x)
                    y = tf.math.add(y, 0.0742610)
                    y = tf.math.multiply(y, x)
                    y = tf.math.subtract(y, 0.2121144)
                    y = tf.math.multiply(y, x)
                    y = tf.math.add(y, 1.5707288)
                    y = tf.math.multiply(y, tf.sqrt(tf.math.subtract(1.0, x)))
                    y = tf.math.multiply(y, tf.math.subtract(1.0, tf.math.multiply(2.0, neg)))
                    acos = tf.math.add(tf.math.multiply(neg, 3.14159265358979), y)
                    return acos

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        if output_no_quant_float32_tflite or \
                            output_dynamic_range_quant_tflite or \
                            output_weight_quant_tflite or \
                            output_float16_quant_tflite or \
                            output_integer_quant_tflite or \
                            output_full_integer_quant_tflite or \
                            output_edgetpu or \
                            output_myriad:

                            tf_layers_dict[layer_id] = pseudo_acos(inp, output_myriad)
                        else:
                            tf_layers_dict[layer_id] = tf.math.acos(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        if output_no_quant_float32_tflite or \
                            output_dynamic_range_quant_tflite or \
                            output_weight_quant_tflite or \
                            output_float16_quant_tflite or \
                            output_integer_quant_tflite or \
                            output_full_integer_quant_tflite or \
                            output_edgetpu or \
                            output_myriad:

                            inp = pseudo_acos(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], output_myriad)
                        else:
                            inp = tf.math.acos(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    if output_no_quant_float32_tflite or \
                        output_dynamic_range_quant_tflite or \
                        output_weight_quant_tflite or \
                        output_float16_quant_tflite or \
                        output_integer_quant_tflite or \
                        output_full_integer_quant_tflite or \
                        output_edgetpu or \
                        output_myriad:

                        tf_layers_dict[layer_id] = pseudo_acos(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], output_myriad)
                    else:
                        tf_layers_dict[layer_id] = tf.math.acos(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Acosh
            elif layer.attrib['type'] == 'Acosh':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.math.acosh(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.math.acosh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.math.acosh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Asin
            elif layer.attrib['type'] == 'Asin':
                # https://zenn.dev/pinto0309/articles/8f6df1d2304395
                def pseudo_asin(x, output_myriad):
                    x_abs = None
                    if output_myriad:
                        x_abs = tf.math.sqrt(tf.math.square(x)) # OAK-D (MyriadX) is not compatible with ABS.
                    else:
                        x_abs = tf.abs(x)
                    neg = tf.math.divide(tf.math.multiply(tf.minimum(x, 0), -1), x_abs)
                    x = x_abs
                    y = tf.constant(-0.0187293)
                    y = tf.math.multiply(y, x)
                    y = tf.math.add(y, 0.0742610)
                    y = tf.math.multiply(y, x)
                    y = tf.math.subtract(y, 0.2121144)
                    y = tf.math.multiply(y, x)
                    y = tf.math.add(y, 1.5707288)
                    y = tf.math.subtract(tf.math.multiply(3.14159265358979, 0.5), tf.math.multiply(tf.sqrt(tf.math.subtract(1.0, x)), y))
                    asin = tf.math.subtract(y, tf.math.multiply(tf.math.multiply(2, neg), y))
                    return asin

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        if output_no_quant_float32_tflite or \
                            output_dynamic_range_quant_tflite or \
                            output_weight_quant_tflite or \
                            output_float16_quant_tflite or \
                            output_integer_quant_tflite or \
                            output_full_integer_quant_tflite or \
                            output_edgetpu or \
                            output_myriad:

                            tf_layers_dict[layer_id] = pseudo_asin(inp, output_myriad)
                        else:
                            tf_layers_dict[layer_id] = tf.math.asin(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        if output_no_quant_float32_tflite or \
                            output_dynamic_range_quant_tflite or \
                            output_weight_quant_tflite or \
                            output_float16_quant_tflite or \
                            output_integer_quant_tflite or \
                            output_full_integer_quant_tflite or \
                            output_edgetpu or \
                            output_myriad:

                            inp = pseudo_asin(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], output_myriad)
                        else:
                            inp = tf.math.asin(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    if output_no_quant_float32_tflite or \
                        output_dynamic_range_quant_tflite or \
                        output_weight_quant_tflite or \
                        output_float16_quant_tflite or \
                        output_integer_quant_tflite or \
                        output_full_integer_quant_tflite or \
                        output_edgetpu or \
                        output_myriad:

                        tf_layers_dict[layer_id] = pseudo_asin(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], output_myriad)
                    else:
                        tf_layers_dict[layer_id] = tf.math.asin(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Asinh
            elif layer.attrib['type'] == 'Asinh':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.math.asinh(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.math.asinh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.math.asinh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Atan
            elif layer.attrib['type'] == 'Atan':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.math.atan(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.math.atan(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.math.atan(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Atanh
            elif layer.attrib['type'] == 'Atanh':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.math.atanh(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.math.atanh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.math.atanh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Ceiling
            elif layer.attrib['type'] == 'Ceiling':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.math.ceil(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.math.ceil(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.math.ceil(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Cos
            elif layer.attrib['type'] == 'Cos':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.math.cos(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.math.cos(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.math.cos(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Cosh
            elif layer.attrib['type'] == 'Cosh':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.math.cosh(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.math.cosh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.math.cosh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Sin
            elif layer.attrib['type'] == 'Sin':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.math.sin(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.math.sin(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.math.sin(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Sinh
            elif layer.attrib['type'] == 'Sinh':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.math.sinh(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.math.sinh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
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
                    inp = tf.math.floordiv(x, y)
                else:
                    # divide
                    inp = tf.math.divide(x, y)

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### Erf
            elif layer.attrib['type'] == 'Erf':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.math.erf(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.math.erf(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.math.erf(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Floor
            elif layer.attrib['type'] == 'Floor':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.math.floor(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.math.floor(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.math.floor(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### FloorMod
            elif layer.attrib['type'] == 'FloorMod':
                inp = tf.math.floormod(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### HSwish
            elif layer.attrib['type'] == 'HSwish':
                if replace_swish_and_hardswish:
                    # Swish
                    if wr_config and layer_id in wr_config and format_version >= 2:
                        if wr_config[layer_id]['replace_mode'] == 'insert_before':
                            inp = extrapolation_of_layers(
                                wr_config[layer_id],
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                            )
                            tf_layers_dict[layer_id] = tf.nn.swish(inp)

                        elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                            inp = tf.nn.swish(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                            tf_layers_dict[layer_id] = extrapolation_of_layers(
                                wr_config[layer_id],
                                inp
                            )
                    else:
                        tf_layers_dict[layer_id] = tf.nn.swish(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

                else:
                    multiplier = 0.0
                    if not optimizing_hardswish_for_edgetpu:
                        multiplier = 0.16666667
                    else:
                        multiplier = 0.16666666

                    # Hard-Swish
                    if wr_config and layer_id in wr_config and format_version >= 2:
                        if wr_config[layer_id]['replace_mode'] == 'insert_before':
                            inp = extrapolation_of_layers(
                                wr_config[layer_id],
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                            )
                            tf_layers_dict[layer_id] = inp * tf.nn.relu6(inp + 3) * multiplier

                        elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                            inp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] * \
                                tf.nn.relu6(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] + 3) * multiplier
                            tf_layers_dict[layer_id] = extrapolation_of_layers(
                                wr_config[layer_id],
                                inp
                            )
                    else:
                        tf_layers_dict[layer_id] = \
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] * \
                                tf.nn.relu6(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] + 3) * multiplier

            ### Log
            elif layer.attrib['type'] == 'Log':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.math.log(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.math.log(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.math.log(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Power
            elif layer.attrib['type'] == 'Power':
                # No broadcast
                port0 = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                port1 = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]

                if output_edgetpu and isinstance(port1, np.ndarray) and port1.size == 1:
                    port1 = port1.squeeze()
                    if port1 > 2.0:
                        port1 = float(port1)
                        if is_integer_num(port1):
                            port1 = int(port1) - 1
                            for i in range(port1):
                                if i == 0:
                                    inp = tf.math.multiply(
                                        port0,
                                        (port0 * 1.0)
                                    )
                                else:
                                    inp = tf.math.multiply(
                                        inp,
                                        (port0 * 1.0)
                                    )
                        else:
                            inp = tf.math.pow(
                                port0,
                                port1
                            )
                    else:
                        inp = tf.math.pow(
                            port0,
                            port1
                        )
                else:
                    inp = tf.math.pow(
                        port0,
                        port1
                    )

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### Mish
            elif layer.attrib['type'] == 'Mish':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = inp * tf.math.tanh(tf.math.softplus(inp))

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = \
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] * \
                                tf.math.tanh(tf.math.softplus(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]))
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = \
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] * \
                            tf.math.tanh(tf.math.softplus(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]))

            ### Selu
            elif layer.attrib['type'] == 'Selu':
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.nn.selu(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.nn.selu(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.nn.selu(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Subtract
            elif layer.attrib['type'] == 'Subtract':
                # No broadcast
                x = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                y = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                if type(x) == np.ndarray:
                    x = x.astype(np.float32)
                    if x.ndim == 4:
                        # 4D - NCHW->NHWC
                        x = x.transpose([0,2,3,1])
                if type(y) == np.ndarray:
                    y = y.astype(np.float32)
                    if y.ndim == 4:
                        # 4D - NCHW->NHWC
                        y = y.transpose([0,2,3,1])
                inp = tf.math.subtract(x, y)

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### Unsqueeze - TODO
            elif layer.attrib['type'] == 'Unsqueeze':
                input_shape = np.asarray(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape)
                indices = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]

                # If the Unsqueeze axis has been changed by extrapolation, the extrapolated axis is given priority.
                try:
                    before_layer_id = str(int(layer_id) - 1)
                    if wr_config and before_layer_id in wr_config and format_version >= 2:
                        if wr_config[before_layer_id]['type'] == 'Const':
                            indices = wr_config[before_layer_id]['values']
                except:
                    pass

                if len(input_shape) > 1 and len(indices) > 1:
                    print('The multi-dimensional indices specification in Unsqueeze is not yet implemented.')
                    sys.exit(-1)
                else:
                    inp = tf.expand_dims(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        indices[0]
                    )
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### Equal
            elif layer.attrib['type'] == 'Equal':
                inp = tf.math.equal(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### NotEqual
            elif layer.attrib['type'] == 'NotEqual':
                inp = tf.math.not_equal(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### Greater
            elif layer.attrib['type'] == 'Greater':
                inp = tf.math.greater(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### GreaterEqual
            elif layer.attrib['type'] == 'GreaterEqual':
                inp = tf.math.greater_equal(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### Less
            elif layer.attrib['type'] == 'Less':
                inp = tf.math.less(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### LessEqual
            elif layer.attrib['type'] == 'LessEqual':
                inp = tf.math.less_equal(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### Select
            elif layer.attrib['type'] == 'Select':
                try:
                    inp = tf.raw_ops.SelectV2(
                        condition=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        t=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                        e=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]
                    )
                except:
                    try:
                        inp = tf.where(
                            condition=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            x=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                            y=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]
                        )
                    except:
                        inp = tf.where(
                            condition=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)],
                            x=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                            y=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### LogicalAnd
            elif layer.attrib['type'] == 'LogicalAnd':
                inp = tf.math.logical_and(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### LogicalNot
            elif layer.attrib['type'] == 'LogicalNot':
                inp = tf.math.logical_not(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### LogicalOr
            elif layer.attrib['type'] == 'LogicalOr':
                inp = tf.math.logical_or(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### LogicalXor
            elif layer.attrib['type'] == 'LogicalXor':
                inp = tf.math.logical_xor(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### Broadcast - TODO
            elif layer.attrib['type'] == 'Broadcast':
                mode = data.attrib['mode']
                if type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]) != np.ndarray:
                    if mode == 'numpy':
                        inp = tf.broadcast_to(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                        )
                    elif mode == 'bidirectional':
                        inp = tf.math.multiply(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            tf.ones(
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                                dtype=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].dtype
                            )
                        )
                    else:
                        print(f'The {mode} mode of broadcast is not yet implemented.')
                        sys.exit(-1)
                else:
                    target_shape = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                    if len(target_shape) == 4:
                        target_shape[0], target_shape[1], target_shape[2], target_shape[3] = \
                            target_shape[0], target_shape[2], target_shape[3], target_shape[1]
                        if mode == 'numpy':
                            inp = tf.broadcast_to(
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                target_shape
                            )
                        elif mode == 'bidirectional':
                            inp = tf.math.multiply(
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                tf.ones(target_shape)
                            )
                        else:
                            print(f'{Color.RED}ERROR:{Color.RESET} Broadcast lyaer {mode} mode of broadcast is not yet implemented. layer_id: {layer_id}')
                            sys.exit(-1)
                    else:
                        if mode == 'numpy':
                            inp = tf.broadcast_to(
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                target_shape
                            )
                        elif mode == 'bidirectional':
                            inp = tf.math.multiply(
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                tf.ones(target_shape)
                            )
                        else:
                            print(f'{Color.RED}ERROR:{Color.RESET} Broadcast lyaer {mode} mode of broadcast is not yet implemented. layer_id: {layer_id}')
                            sys.exit(-1)

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### Split
            elif layer.attrib['type'] == 'Split':
                num_splits = int(data.attrib['num_splits'])
                inp_shape_len = len(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape)
                axis = int(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])
                if inp_shape_len >= 4:
                    if axis == 1:
                        axis = 3
                    elif axis >= 2:
                        axis -= 1
                else:
                    axis = axis

                def split_tensor(x, axis, num_split):
                    return tf.raw_ops.Split(axis=axis, value=x, num_split=num_split)

                outputs = Lambda(
                    split_tensor,
                    arguments={
                        'axis': axis,
                        'num_split': num_splits}
                    )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

                for output, layer_id_port in zip(outputs, layer_id_port_dict[layer_id]['layer_id:port']):
                    tf_layers_dict[layer_id_port] = output

                if wr_config and layer_id in wr_config and format_version >= 2:
                    print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to "Split" is not supported. layer_id: {layer_id}')
                    sys.exit(-1)

            ### VariadicSplit
            elif layer.attrib['type'] == 'VariadicSplit':
                axis = int(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])
                input_shape_len = len(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape)

                if input_shape_len == 4:
                    if axis == 1:
                        axis = -1
                    elif axis >= 2:
                        axis -= 1
                else:
                    pass

                num_or_size_splits = None
                if type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]) == np.ndarray:
                    num_or_size_splits = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]
                else:
                    num_or_size_splits = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]

                def split_tensor(x, axis, num_split):
                    return tf.raw_ops.Split(axis=axis, value=x, num_split=num_split)

                if len(num_or_size_splits) > 1 and np.average(num_or_size_splits) == num_or_size_splits[0]:
                    outputs = Lambda(
                        split_tensor,
                        arguments={
                            'axis': axis,
                            'num_split': len(num_or_size_splits)}
                        )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                else:
                    if len(num_or_size_splits) > 1:
                        outputs = []

                        start_idx = 0
                        end_idx   = 0
                        split_start_list_all = []

                        for split_len in num_or_size_splits:
                            split_start_list_part = []
                            end_idx = start_idx + split_len - 1
                            for i in range(input_shape_len):
                                if axis == -1 and i == (input_shape_len - 1):
                                    split_start_list_part.append(start_idx)
                                elif axis == -1 and i != (input_shape_len - 1):
                                    split_start_list_part.append(0)
                                elif axis != -1 and i == axis:
                                    split_start_list_part.append(start_idx)
                                else:
                                    split_start_list_part.append(0)
                            split_start_list_all.append(split_start_list_part)
                            start_idx = end_idx + 1

                        split_size_list_all = []

                        for split_len in num_or_size_splits:
                            split_size_list_part = []
                            for i in range(input_shape_len):
                                if axis == -1 and i == (input_shape_len - 1):
                                    split_size_list_part.append(split_len)
                                elif axis == -1 and i != (input_shape_len - 1):
                                    split_size_list_part.append(-1)
                                elif axis != -1 and i == axis:
                                    split_size_list_part.append(split_len)
                                else:
                                    split_size_list_part.append(-1)
                            split_size_list_all.append(split_size_list_part)

                        for split_starts, split_sizes in zip(split_start_list_all, split_size_list_all):
                            outputs.append(
                                tf.slice(
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                    split_starts, split_sizes
                                )
                            )

                    else:
                        outputs = tf.split(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            num_or_size_splits=num_or_size_splits,
                            axis=axis
                        )

                for output, layer_id_port in zip(outputs, layer_id_port_dict[layer_id]['layer_id:port']):
                    tf_layers_dict[layer_id_port] = output

                if wr_config and layer_id in wr_config and format_version >= 2:
                    print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to "VariadicSplit" is not supported. layer_id: {layer_id}')
                    sys.exit(-1)

            ### MVN - TODO axes
            elif layer.attrib['type'] == 'MVN':
                eps = float(data.attrib['eps'])
                across_channels = None
                if not data is None and 'across_channels' in data.attrib:
                    across_channels = data.attrib['across_channels']
                normalize_variance = None
                if not data is None and 'normalize_variance' in data.attrib:
                    normalize_variance = data.attrib['normalize_variance']
                eps_mode = None
                if not data is None and 'eps_mode' in data.attrib:
                    eps_mode = data.attrib['eps_mode'].lower()

                if across_channels == '0':
                    across_channels = False
                elif across_channels == '1':
                    across_channels = True
                elif across_channels == 'False' or across_channels == 'false':
                    across_channels = False
                elif across_channels == 'True' or across_channels == 'true':
                    across_channels = True

                if normalize_variance == '0':
                    normalize_variance = False
                elif normalize_variance == '1':
                    normalize_variance = True
                elif normalize_variance == 'False' or normalize_variance == 'false':
                    normalize_variance = False
                elif normalize_variance == 'True' or normalize_variance == 'true':
                    normalize_variance = True

                mean = None
                var = None
                axes = None
                data = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                try:
                    axes = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                    if len(data.shape) == 4:
                        # NCHW : NHWC
                        if type(axes) == np.ndarray and len(axes.shape) == 1 and axes == [1]:
                            axes = 3
                        elif type(axes) == np.ndarray and len(axes.shape) == 1 and axes == [2] or axes == [3]:
                            axes = axes - 1
                        else:
                            axes = axes
                    elif len(data.shape) == 5:
                        # NCDHW : NDHWC
                        if type(axes) == np.ndarray and len(axes.shape) == 1 and axes == [1]:
                            axes = 4
                        elif type(axes) == np.ndarray and len(axes.shape) == 1 and axes == [2] or axes == [3] or axes == [4]:
                            axes = axes - 1
                        else:
                            axes = axes
                        print(f'{Color.YELLOW}WARNING:{Color.RESET} It is possible that the MVN axes are not accurately predicted. Check if the dimension of the transformed Mean operation axis is correct. layer_id: {layer_id}')
                    else:
                        # 2D, 3D, 6D+
                        axes = axes
                        print(f'{Color.YELLOW}WARNING:{Color.RESET} It is possible that the MVN axes are not accurately predicted. Check if the dimension of the transformed Mean operation axis is correct. layer_id: {layer_id}')
                except:
                    pass

                if across_channels:
                    mean = tf.math.reduce_mean(data, axis=tf.constant([-1], dtype=tf.int32), keepdims=True)
                    var = tf.math.reduce_variance(data, axis=tf.constant([-1], dtype=tf.int32), keepdims=True)
                else:
                    mean = tf.math.reduce_mean(data, axis=tf.cast(axes, dtype=tf.int32), keepdims=True)
                    var = tf.math.reduce_variance(data, axis=tf.cast(axes, dtype=tf.int32), keepdims=True)

                if normalize_variance:
                    if eps_mode is not None and eps_mode == 'inside_sqrt':
                        mvn = (data - mean) / tf.math.sqrt(var + eps)
                    elif eps_mode is not None and eps_mode == 'outside_sqrt':
                        mvn = (data - mean) / (tf.math.sqrt(var) + eps)
                    else:
                        mvn = (data - mean) / tf.math.sqrt(var + eps)
                else:
                    mvn = (data - mean)

                inp = mvn

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### NonMaxSuppression - TODO
            elif layer.attrib['type'] == 'NonMaxSuppression':
                boxes = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)][0]
                batch_size = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape[0]
                if batch_size > 1:
                    print('When using NonMaxSuppression, fix the batch size to 1.')
                    sys.exit(-1)
                total_boxes_count = boxes.shape[0]
                scores = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)][0]
                class_count = scores.shape[0]
                if class_count > 1:
                    print('When using NonMaxSuppression, fix the class_count to 1. Forcibly reduces the number of classes to 1.')

                selected_boxes = None
                selected_class_idx = None

                max_output_boxes_per_class = 0
                try:
                    max_output_boxes_per_class = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)][0] # 25
                    if max_output_boxes_per_class == '-inf' or \
                        max_output_boxes_per_class == 'inf' or \
                            max_output_boxes_per_class == '-Infinity' or \
                                max_output_boxes_per_class == 'Infinity':
                        max_output_boxes_per_class = 0
                    elif max_output_boxes_per_class == 9223372036854775807:
                        max_output_boxes_per_class = total_boxes_count
                except:
                    pass

                iou_threshold = np.asarray(0.0, dtype=np.float32)
                try:
                    iou_threshold = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 3)][0] # 0.5
                    if iou_threshold == '-inf' or \
                        iou_threshold == 'inf' or \
                            iou_threshold == '-Infinity' or \
                                iou_threshold == 'Infinity':
                        iou_threshold = np.asarray(0.0, dtype=np.float32)
                    if type(iou_threshold) is np.float64:
                        iou_threshold = np.asarray(iou_threshold, dtype=np.float32)
                except:
                    pass

                score_threshold = np.asarray(0.0, dtype=np.float32)
                try:
                    score_threshold = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 4)][0] # -Infinity
                    if score_threshold == '-inf' or \
                        score_threshold == 'inf' or \
                            score_threshold == '-Infinity' or \
                                score_threshold == 'Infinity' or \
                                    score_threshold == float('-inf'):
                        score_threshold = np.asarray(0.0, dtype=np.float32)
                    if type(score_threshold) is np.float64:
                        score_threshold = np.asarray(score_threshold, dtype=np.float32)
                except:
                    pass

                soft_nms_sigma = np.asarray(0.0, dtype=np.float32)
                try:
                    soft_nms_sigma = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 5) ][0] # 0.1
                    if soft_nms_sigma == '-inf' or \
                        soft_nms_sigma == 'inf' or \
                            soft_nms_sigma == '-Infinity' or \
                                soft_nms_sigma == 'Infinity':
                        soft_nms_sigma = np.asarray(0.0, dtype=np.float32)
                except:
                    pass
                if type(soft_nms_sigma) is np.float64:
                    soft_nms_sigma = np.asarray(soft_nms_sigma, dtype=np.float32)
                box_encoding = data.attrib['box_encoding'] # corner or center
                output_type = data.attrib['output_type'] # i64 or i32
                sort_result_descending_str = data.attrib['sort_result_descending'] # True/true or False/false
                sort_result_descending = False
                if sort_result_descending_str == 'True' or sort_result_descending_str == 'true':
                    sort_result_descending = True

                if box_encoding == 'corner':

                    scores_tmp = tf.transpose(scores, perm=[1, 0])
                    score_top_values, score_top_idxes = tf.math.top_k(input=scores_tmp, k=1, sorted=False)
                    score_top_values_flat = tf.reshape(score_top_values, [-1])
                    score_top_idxes_flat = tf.reshape(score_top_idxes, [-1])
                    output_size = tf.math.minimum(total_boxes_count, max_output_boxes_per_class)

                    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
                        boxes=boxes,
                        scores=score_top_values_flat,
                        max_output_size=output_size,
                        iou_threshold=iou_threshold,
                        score_threshold=score_threshold,
                        soft_nms_sigma=soft_nms_sigma
                    )
                    selected_boxes = tf.gather(boxes, selected_indices)
                    selected_class_idx = tf.gather(score_top_idxes_flat, selected_indices)

                elif box_encoding == 'center':
                    print(f'The NonMaxSuppression box_encoding=center mode is not yet implemented.')
                    sys.exit(-1)

                negative_value_template = tf.fill([output_size], -1)

                selected_class_idx_shape = tf.shape(selected_class_idx)[0]
                selected_class_idx_range = tf.range(selected_class_idx_shape)
                selected_class_idx_indices = tf.reshape(selected_class_idx_range, (selected_class_idx_shape, 1))

                selected_boxes_shape = tf.shape(selected_boxes)[0]
                selected_boxes_range = tf.range(selected_boxes_shape)
                selected_boxes_indices = tf.reshape(selected_boxes_range, (selected_boxes_shape, 1))

                selected_scores_shape = tf.shape(selected_scores)[0]
                selected_scores_range = tf.range(selected_scores_shape)
                selected_scores_indices = tf.reshape(selected_scores_range, (selected_scores_shape, 1))

                selected_class_indices_upd = tf.tensor_scatter_nd_update(
                    negative_value_template,
                    selected_class_idx_indices,
                    selected_class_idx
                )

                # selected_indices
                selected_box_indices_upd = tf.tensor_scatter_nd_update(
                    negative_value_template,
                    selected_boxes_indices,
                    tf.cast(selected_indices, dtype=tf.int32)
                )
                selected_indices_vino = tf.stack(
                    [
                        negative_value_template,
                        selected_class_indices_upd,
                        selected_box_indices_upd
                    ]
                    , axis=1
                )

                # selected_scores
                selected_scores_upd = tf.tensor_scatter_nd_update(
                    tf.cast(negative_value_template, dtype=tf.float32),
                    selected_scores_indices,
                    selected_scores
                )
                selected_scores_vino = tf.stack(
                    [
                        tf.cast(negative_value_template, dtype=tf.float32),
                        tf.cast(selected_class_indices_upd, dtype=tf.float32),
                        selected_scores_upd
                    ]
                    , axis=1
                )

                # valid_outputs
                valid_outputs_vino = total_boxes_count

                outputs = [selected_indices_vino, selected_scores_vino, valid_outputs_vino]
                for output, layer_id_port in zip(outputs, layer_id_port_dict[layer_id]['layer_id:port']):
                    tf_layers_dict[layer_id_port] = output

                if wr_config and layer_id in wr_config and format_version >= 2:
                    print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to "NonMaxSuppression" is not supported. layer_id: {layer_id}')
                    sys.exit(-1)

            ### NonZero
            elif layer.attrib['type'] == 'NonZero':
                output_type = tf.int64
                if not data is None and 'output_type' in data.attrib:
                    output_type = cast_type_ov_tf[data.attrib['output_type']]

                try:
                    # type_spec.dtype
                    if tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].type_spec.dtype != tf.bool:
                        input_type = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].type_spec.dtype
                        input_shape = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape
                        mask = tf.math.not_equal(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            tf.zeros(input_shape, dtype=input_type)
                        )
                        inp = tf.expand_dims(
                            tf.boolean_mask(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], mask),
                            axis=0
                        )
                    else:
                        temp_op = tf.cast(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], output_type)
                        mask = tf.math.not_equal(
                            temp_op,
                            tf.zeros(temp_op.shape, dtype=output_type)
                        )
                        inp = tf.expand_dims(
                            tf.boolean_mask(temp_op, mask),
                            axis=0
                        )
                except:
                    # dtype
                    if tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].dtype != tf.bool:
                        input_type = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].dtype
                        input_shape = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape
                        mask = tf.math.not_equal(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            tf.zeros(input_shape, dtype=input_type)
                        )
                        inp = tf.expand_dims(
                            tf.boolean_mask(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], mask),
                            axis=0
                        )
                    else:
                        temp_op = tf.cast(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], output_type)
                        mask = tf.math.not_equal(
                            temp_op,
                            tf.zeros(temp_op.shape, dtype=output_type)
                        )
                        inp = tf.expand_dims(
                            tf.boolean_mask(temp_op, mask),
                            axis=0
                        )

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### SpaceToDepth
            elif layer.attrib['type'] == 'SpaceToDepth':
                block_size = int(data.attrib['block_size'])
                mode = data.attrib['mode']

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = tf.nn.space_to_depth(
                            inp,
                            block_size=block_size
                        )

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.nn.space_to_depth(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            block_size=block_size
                        )
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = tf.nn.space_to_depth(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        block_size=block_size
                    )

            ### DepthToSpace
            elif layer.attrib['type'] == 'DepthToSpace':
                block_size = int(data.attrib['block_size'])
                mode = data.attrib['mode']

                def depth_to_space(x, block_size):
                    return tf.raw_ops.DepthToSpace(input=x, block_size=block_size)

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                        tf_layers_dict[layer_id] = Lambda(
                            depth_to_space,
                            arguments={'block_size': block_size}
                        )(inp)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = Lambda(
                            depth_to_space,
                            arguments={'block_size': block_size}
                        )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = Lambda(
                        depth_to_space,
                        arguments={'block_size': block_size}
                    )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Sqrt
            elif layer.attrib['type'] == 'Sqrt':
                inp = tf.math.sqrt(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                )
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### SquaredDifference
            elif layer.attrib['type'] == 'SquaredDifference':
                inp = tf.math.squared_difference(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### FakeQuantize
            elif layer.attrib['type'] == 'FakeQuantize':
                x = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                x = tf.cast(x, dtype=tf.float32)
                if isinstance(x, tf.Tensor) and len(x.shape) == 4:
                    x = x.numpy()
                elif type(x) == np.ndarray and len(x.shape) == 4:
                    x = x

                input_low = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                input_low = tf.cast(input_low, dtype=tf.float32)
                if isinstance(input_low, tf.Tensor) and len(input_low.shape) == 4:
                    input_low = input_low.numpy().transpose(0,2,3,1)
                elif type(input_low) == np.ndarray and len(input_low.shape) == 4:
                    input_low = input_low.transpose(0,2,3,1)

                input_high = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]
                input_high = tf.cast(input_high, dtype=tf.float32)
                if isinstance(input_high, tf.Tensor) and len(input_high.shape) == 4:
                    input_high = input_high.numpy().transpose(0,2,3,1)
                elif type(input_high) == np.ndarray and len(input_high.shape) == 4:
                    input_high = input_high.transpose(0,2,3,1)

                output_low = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 3)]
                output_low = tf.cast(output_low, dtype=tf.float32)
                if isinstance(output_low, tf.Tensor) and len(output_low.shape) == 4:
                    output_low = output_low.numpy().transpose(0,2,3,1)
                elif type(output_low) == np.ndarray and len(output_low.shape) == 4:
                    output_low = output_low.transpose(0,2,3,1)

                output_high = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 4)]
                output_high = tf.cast(output_high, dtype=tf.float32)
                if isinstance(output_high, tf.Tensor) and len(output_high.shape) == 4:
                    output_high = output_high.numpy().transpose(0,2,3,1)
                elif type(output_high) == np.ndarray and len(output_high.shape) == 4:
                    output_high = output_high.transpose(0,2,3,1)

                levels = int(data.attrib['levels'])

                ### https://stackoverflow.com/questions/64111110/how-to-do-round-half-up-in-tensorflow
                ### https://www.xspdf.com/resolution/50434452.html
                inp = tf.where(tf.math.less_equal(x, tf.math.minimum(input_low, input_high)), output_low,
                                                    tf.where(tf.math.greater(x, tf.math.maximum(input_low, input_high)), output_high,
                                                    tf.floor(((x - input_low) / (input_high - input_low) * (levels-1)) + 0.5) / (levels-1) * (output_high - output_low) + output_low))

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### Tile
            elif layer.attrib['type'] == 'Tile':
                inp = tf.tile(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### Gelu
            elif layer.attrib['type'] == 'Gelu':
                approximation_mode = None
                if not data is None and 'approximation_mode' in data.attrib:
                    approximation_mode = data.attrib['approximation_mode']
                if approximation_mode == 'ERF' or approximation_mode is None:
                    inp = tf.nn.gelu(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        approximate=False
                    )
                elif approximation_mode == 'TANH':
                    inp = tf.nn.gelu(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        approximate=True
                    )
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### NormalizeL2
            elif layer.attrib['type'] == 'NormalizeL2':
                eps = 1e-12 #None
                if not data is None and 'eps' in data.attrib:
                    eps = np.array(data.attrib['eps'], dtype=np.float32)
                eps_mode = None
                if not data is None and 'eps_mode' in data.attrib:
                    eps_mode = data.attrib['eps_mode']

                if eps_mode == 'add':
                    inp = tf.math.l2_normalize(
                        tf.math.add(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], eps),
                        axis=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                        epsilon=0.0
                    )
                elif eps_mode == 'max':
                    inp = tf.math.l2_normalize(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        axis=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                        epsilon=eps
                    )
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### ScatterElementsUpdate - WIP
            ### https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/handlers/backend/scatter_elements.py
            elif layer.attrib['type'] == 'ScatterElementsUpdate':
                data = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                indices = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                updates = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]
                axis = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 3)]
                axis = axis if axis >= 0 else tf.add(tf.rank(data), axis)
                data_shape = tf.shape(data)
                max_i = tf.cast(data_shape[axis[0]], indices.dtype)
                indices = tf.math.floormod(tf.add(indices, max_i), max_i)
                sparsified_dense_idx_shape = tf.shape(updates)
                updates_rank = tf.rank(updates)
                try:
                    idx_tensors_per_axis = [tf.range(sparsified_dense_idx_shape[i]) for i in range(updates_rank)]
                except:
                    idx_tensors_per_axis = [tf.range(sparsified_dense_idx_shape[i]) for i in range(1)]
                idx_tensors_per_axis = tf.meshgrid(*idx_tensors_per_axis, indexing='ij')
                idx_tensors_per_axis[int(axis)] = indices
                dim_expanded_idx_tensors_per_axis = [tf.expand_dims(idx_tensor, axis=-1) for idx_tensor in idx_tensors_per_axis]
                coordinate = tf.concat(dim_expanded_idx_tensors_per_axis, axis=-1)
                indices = tf.reshape(coordinate, [-1, tf.rank(data)])
                updates = tf.reshape(updates, [-1])
                inp = tf.tensor_scatter_nd_update(data, indices, updates)

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### ScatterNDUpdate - WIP
            elif layer.attrib['type'] == 'ScatterNDUpdate':
                data = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                indices = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                updates = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]
                inp = tf.tensor_scatter_nd_update(data, indices, updates)

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### ROIAlign - WIP
            ### https://github.com/tensorpack/tensorpack/blob/3e0dffaccdb6a36490970014570b5e7426b55bf5/examples/FasterRCNN/model_box.py#L156-L172
            elif layer.attrib['type'] == 'ROIAlign':
                mode = None
                pooled_h = None
                pooled_w = None
                sampling_ratio = None
                spatial_scale = None
                if not data is None and 'mode' in data.attrib:
                    mode = data.attrib['mode']
                if not data is None and 'pooled_h' in data.attrib:
                    pooled_h = int(data.attrib['pooled_h'])
                if not data is None and 'pooled_w' in data.attrib:
                    pooled_w = int(data.attrib['pooled_w'])
                if not data is None and 'sampling_ratio' in data.attrib:
                    sampling_ratio = data.attrib['sampling_ratio']
                if not data is None and 'spatial_scale' in data.attrib:
                    spatial_scale = data.attrib['spatial_scale']
                image = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                boxes = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                box_indices = tf.cast(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)], tf.int32)

                # roi_width = max(spatial_scale * (x_2 - x_1), 1.0)
                # roi_height = max(spatial_scale * (y_2 - y_1), 1.0)

                crop_and_resize = tf.image.crop_and_resize(
                    image=image,
                    boxes=boxes,
                    # box_indices=tf.zeros([tf.shape(boxes)[0]], dtype=tf.int32),
                    box_indices=box_indices,
                    crop_size=[pooled_h*2, pooled_w*2],
                    method='bilinear',
                    extrapolation_value=0
                )
                inp = None
                if mode == 'avg':
                    inp = tf.nn.avg_pool(
                        input=crop_and_resize,
                        ksize=[1,2,2,1],
                        strides=[1,2,2,1],
                        padding='SAME'
                    )
                elif mode == 'max':
                    inp = tf.nn.max_pool(
                        input=crop_and_resize,
                        ksize=[1,2,2,1],
                        strides=[1,2,2,1],
                        padding='SAME'
                    )
                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### GatherElements - WIP
            ### https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/handlers/backend/gather_elements.py
            elif layer.attrib['type'] == 'GatherElements':
                axis = None
                if not data is None and 'axis' in data.attrib:
                    axis = int(data.attrib['axis'])

                data = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                indices = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                axis = axis if axis >= 0 else tf.add(tf.rank(data), axis)
                data_shape = tf.shape(data)
                max_i = tf.cast(data_shape[axis], indices.dtype)
                indices = tf.math.floormod(tf.add(indices, max_i), max_i)

                if axis == 0:
                    axis_perm = tf.range(tf.rank(data))
                    data_swaped = data
                    index_swaped = indices
                else:
                    axis_perm = tf.tensor_scatter_nd_update(
                        tf.range(tf.rank(data)),
                        tf.constant([[0], [axis]]),
                        tf.constant([axis, 0])
                    )
                    data_swaped = tf.transpose(data, perm=axis_perm)
                    index_swaped = tf.transpose(indices, perm=axis_perm)

                idx_tensors_per_axis = [
                    tf.range(tf.shape(index_swaped, index_swaped.dtype)[i])
                    for i in range(index_swaped.shape.rank)
                ]
                idx_tensors_per_axis = tf.meshgrid(*idx_tensors_per_axis, indexing='ij')
                idx_tensors_per_axis[0] = index_swaped
                dim_expanded_idx_tensors_per_axis = [
                    tf.expand_dims(idx_tensor, axis=-1)
                    for idx_tensor in idx_tensors_per_axis
                ]
                index_expanded = tf.concat(dim_expanded_idx_tensors_per_axis, axis=-1)

                gathered = tf.gather_nd(data_swaped, index_expanded)
                inp = tf.transpose(gathered, perm=axis_perm)

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### ConvertLike
            elif layer.attrib['type'] == 'ConvertLike':
                port1 = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                port2 = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                like = port2.dtype

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            port1
                        )
                        tf_layers_dict[layer_id] = tf.cast(
                            inp,
                            like
                        )

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.cast(
                            port1,
                            like
                        )
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )

                else:
                    tf_layers_dict[layer_id] = tf.cast(
                        port1,
                        like
                    )

            ### ShuffleChannels
            elif layer.attrib['type'] == 'ShuffleChannels':
                port1 = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]

                axis = None
                if not data is None and 'axis' in data.attrib:
                    axis = int(data.attrib['axis'])

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['type'] == 'ShuffleChannels' and wr_config[layer_id]['replace_mode'] == 'change_axis':
                        axis = int(wr_config[layer_id]['values'])

                if axis != 1 or len(np.asarray(port1.shape)) != 4:
                    print('Operations other than axis=1 for ShuffleChannels or tensor dimensions other than 4 dimensions are not implemented.')
                    sys.exit(-1)

                if axis == 1 and len(np.asarray(port1.shape)) == 4:
                    axis = -1
                elif (axis == -1 or axis == len(np.asarray(port1.shape)) - 1) and \
                    len(np.asarray(port1.shape)) == 4:
                    axis = 1
                elif len(np.asarray(port1.shape)) < 4:
                    pass
                elif axis > 0:
                    pass

                group = 1
                if not data is None and 'group' in data.attrib:
                    group = int(data.attrib['group'])

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            port1
                        )
                        shufflechannels_reshape = tf.reshape(
                            inp,
                            [inp.shape[0], inp.shape[1] * inp.shape[2], group, inp.shape[3] // group]
                        )
                        shufflechannels_transpose = tf.transpose(
                            shufflechannels_reshape,
                            perm=[0, 1, 3, 2]
                        )
                        tf_layers_dict[layer_id] = tf.reshape(
                            shufflechannels_transpose,
                            [inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]]
                        )

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        shufflechannels_reshape = tf.reshape(
                            port1,
                            [port1.shape[0], port1.shape[1] * port1.shape[2], group, port1.shape[3] // group]
                        )
                        shufflechannels_transpose = tf.transpose(
                            shufflechannels_reshape,
                            perm=[0, 1, 3, 2]
                        )
                        inp = tf.reshape(
                            shufflechannels_transpose,
                            [port1.shape[0], port1.shape[1], port1.shape[2], port1.shape[3]]
                        )
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                    else:
                        shufflechannels_reshape = tf.reshape(
                            port1,
                            [port1.shape[0], port1.shape[1] * port1.shape[2], group, port1.shape[3] // group]
                        )
                        shufflechannels_transpose = tf.transpose(
                            shufflechannels_reshape,
                            perm=[0, 1, 3, 2]
                        )
                        tf_layers_dict[layer_id] = tf.reshape(
                            shufflechannels_transpose,
                            [port1.shape[0], port1.shape[1], port1.shape[2], port1.shape[3]]
                        )
                else:
                    shufflechannels_reshape = tf.reshape(
                        port1,
                        [port1.shape[0], port1.shape[1] * port1.shape[2], group, port1.shape[3] // group]
                    )
                    shufflechannels_transpose = tf.transpose(
                        shufflechannels_reshape,
                        perm=[0, 1, 3, 2]
                    )
                    tf_layers_dict[layer_id] = tf.reshape(
                        shufflechannels_transpose,
                        [port1.shape[0], port1.shape[1], port1.shape[2], port1.shape[3]]
                    )

            ### PriorBox
            elif layer.attrib['type'] == 'PriorBox':
                aspect_ratio = None
                if not data is None and 'aspect_ratio' in data.attrib:
                    aspect_ratio = np.asarray(data.attrib['aspect_ratio'].replace(' ', '').split(','), dtype=np.float32)
                clip = False
                if not data is None and 'clip' in data.attrib:
                    clip = data.attrib['clip']
                    clip = True if clip.lower() == 'true' or clip == "1" else False
                densities = []
                if not data is None and 'density' in data.attrib and data.attrib['density'].replace(' ', ''):
                    densities = np.asarray(data.attrib['density'].replace(' ', '').split(','), dtype=np.float32)
                fixed_ratio = []
                if not data is None and 'fixed_ratio' in data.attrib and data.attrib['fixed_ratio'].replace(' ', ''):
                    fixed_ratio = np.asarray(data.attrib['fixed_ratio'].replace(' ', '').split(','), dtype=np.float32)
                fixed_size = []
                if not data is None and 'fixed_size' in data.attrib and data.attrib['fixed_size'].replace(' ', ''):
                    fixed_size = np.asarray(data.attrib['fixed_size'].replace(' ', '').split(','), dtype=np.float32)
                flip = False
                if not data is None and 'flip' in data.attrib:
                    flip = data.attrib['flip']
                    flip = True if flip.lower() == 'true' or flip == "1" else False
                max_size = []
                if not data is None and 'max_size' in data.attrib:
                    max_size = np.asarray(data.attrib['max_size'].replace(' ', '').split(','), dtype=np.float32)
                min_size = []
                if not data is None and 'min_size' in data.attrib:
                    min_size = np.asarray(data.attrib['min_size'].replace(' ', '').split(','), dtype=np.float32)
                offset = None
                if not data is None and 'offset' in data.attrib:
                    offset = np.asarray(data.attrib['offset'], dtype=np.float32)
                scale_all_sizes = True
                if not data is None and 'scale_all_sizes' in data.attrib:
                    scale_all_sizes = data.attrib['scale_all_sizes']
                    scale_all_sizes = True if scale_all_sizes.lower() == 'true' else False
                step = 0.0
                if not data is None and 'step' in data.attrib:
                    step = np.asarray(data.attrib['step'], dtype=np.float32)
                variance = None
                if not data is None and 'variance' in data.attrib:
                    variance = np.asarray(data.attrib['variance'].replace(' ', '').split(','), dtype=np.float32)

                port1 = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] # output_size
                port2 = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)] # image_size

                layer_width = port1[1]
                layer_height = port1[0]
                img_width = port2[1]
                img_height = port2[0]

                def normalized_aspect_ratio(aspect_ratio, flip):
                    unique_ratios = []
                    for ratio in aspect_ratio:
                        val = (round(ratio * 1e6) / 1e6)
                        if val not in unique_ratios:
                            unique_ratios.append(val)
                        if flip:
                            val = (round(1 / ratio * 1e6) / 1e6)
                            if val not in unique_ratios:
                                unique_ratios.append(val)
                    unique_ratios.append(1.0)
                    return unique_ratios

                num_priors = 0
                total_aspect_ratios = len(normalized_aspect_ratio(aspect_ratio, flip))
                if scale_all_sizes:
                    num_priors = total_aspect_ratios * len(min_size) + len(max_size)
                else:
                    num_priors = total_aspect_ratios + len(min_size) - 1

                if fixed_size:
                    num_priors = total_aspect_ratios * len(fixed_size)

                for density in densities:
                    rounded_density = int(density)
                    density_2d = (rounded_density * rounded_density - 1)
                    if fixed_ratio:
                        num_priors += len(fixed_ratio) * density_2d
                    else:
                        num_priors += total_aspect_ratios * density_2d

                out_shape = [2, 4 * layer_height * layer_width * num_priors]
                dst_data = np.zeros((out_shape), dtype=np.float32).flatten()

                aspect_ratios = [1.0]
                for ratio in aspect_ratio:
                    exist = False
                    for existed_value in aspect_ratios:
                        exist = exist or (abs(ratio - existed_value) < 1e-6)
                    if not exist:
                        aspect_ratios.append(ratio)
                        if flip:
                            aspect_ratios.append(1.0 / ratio)

                if variance is None or len(variance) == 0:
                    variance = [0.1]

                if not scale_all_sizes:
                    if step == -1.0:
                        step = 1.0 * img_height / layer_height
                    else:
                        step *= img_height
                    for size in min_size:
                        size *= img_height

                idx = 0
                IWI = tf.constant([1], dtype=tf.int64) / img_width
                IHI = tf.constant([1], dtype=tf.int64) / img_height
                if step == 0:
                    step_x = img_width / layer_width
                    step_y = img_height / layer_height
                else:
                    step_x = step
                    step_y = step

                def clip_great(x, threshold):
                    if x < threshold:
                        return x
                    else:
                        return threshold

                def clip_less(x, threshold):
                    if x > threshold:
                        return x
                    else:
                        return threshold

                def calculate_data(dst_data, idx, center_x, center_y, box_width, box_height, clip):
                    if clip:
                        # order: xmin, ymin, xmax, ymax
                        dst_data[idx] = clip_less((center_x - box_width) * IWI, 0)
                        idx += 1
                        dst_data[idx] = clip_less((center_y - box_height) * IHI, 0)
                        idx += 1
                        dst_data[idx] = clip_great((center_x + box_width) * IWI, 1)
                        idx += 1
                        dst_data[idx] = clip_great((center_y + box_height) * IHI, 1)
                        idx += 1
                    else:
                        dst_data[idx] = (center_x - box_width) * IWI
                        idx += 1
                        dst_data[idx] = (center_y - box_height) * IHI
                        idx += 1
                        dst_data[idx] = (center_x + box_width) * IWI
                        idx += 1
                        dst_data[idx] = (center_y + box_height) * IHI
                        idx += 1

                for h in range(layer_height):
                    for w in range(layer_width):
                        if step == 0:
                            center_x = (w + 0.5) * step_x
                            center_y = (h + 0.5) * step_y
                        else:
                            center_x = (offset + w) * step
                            center_y = (offset + h) * step

                        for s in range(len(fixed_size)):
                            fixed_size_ = fixed_size[s]
                            box_width = fixed_size_ * 0.5
                            box_height = fixed_size_ * 0.5

                            if not fixed_ratio is None and len(fixed_ratio) > 0:
                                for ar in fixed_ratio:
                                    density_ = int(density[s])
                                    shift = int(fixed_size[s] / density_)
                                    ar = math.sqrt(ar)
                                    box_width_ratio = fixed_size[s] * 0.5 * ar
                                    box_height_ratio = fixed_size[s] * 0.5 / ar
                                    for r in range(density_):
                                        for c in range(density_):
                                            center_x_temp = center_x - fixed_size_ / 2 + shift / 2.0 + c * shift
                                            center_y_temp = center_y - fixed_size_ / 2 + shift / 2.0 + r * shift
                                            calculate_data(idx, center_x_temp, center_y_temp, box_width_ratio, box_height_ratio, True)
                            else:
                                if not density is None and len(density) > 0:
                                    density_ = int(density[s])
                                    shift = int(fixed_size[s] / density_)
                                    for r in range(density_):
                                        for c in range(density_):
                                            center_x_temp = center_x - fixed_size_ / 2 + shift / 2.0 + c * shift
                                            center_y_temp = center_y - fixed_size_ / 2 + shift / 2.0 + r * shift
                                            calculate_data(dst_data, idx, center_x_temp, center_y_temp, box_width, box_height, True)
                                # Rest of priors
                                for ar in aspect_ratios:
                                    if (abs(ar - 1.0) < 1e-6):
                                        continue

                                    density_ = int(density[s])
                                    shift = int(fixed_size[s] / density_)
                                    ar = math.sqrt(ar)
                                    box_width_ratio = fixed_size[s] * 0.5 * ar
                                    box_height_ratio = fixed_size[s] * 0.5 / ar
                                    for r in range(density_):
                                        for c in range(density_):
                                            center_x_temp = center_x - fixed_size_ / 2 + shift / 2.0 + c * shift
                                            center_y_temp = center_y - fixed_size_ / 2 + shift / 2.0 + r * shift
                                            calculate_data(dst_data, idx, center_x_temp, center_y_temp, box_width_ratio, box_height_ratio, True)

                        for ms_idx in range(len(min_size)):
                            box_width = min_size[ms_idx] * 0.5
                            box_height = min_size[ms_idx] * 0.5
                            calculate_data(dst_data, idx, center_x, center_y, box_width, box_height, False)

                            if len(max_size) > ms_idx:
                                box_width = math.sqrt(min_size[ms_idx] * max_size[ms_idx]) * 0.5
                                box_height = math.sqrt(min_size[ms_idx] * max_size[ms_idx]) * 0.5
                                calculate_data(dst_data, idx, center_x, center_y, box_width, box_height, False)

                            if scale_all_sizes or (not scale_all_sizes and (ms_idx == len(min_size) - 1)):
                                s_idx = 0
                                if scale_all_sizes:
                                    s_idx = ms_idx
                                else:
                                    s_idx = 0
                                for ar in aspect_ratios:
                                    if (abs(ar - 1.0) < 1e-6):
                                        continue
                                    ar = math.sqrt(ar)
                                    box_width = min_size[s_idx] * 0.5 * ar
                                    box_height = min_size[s_idx] * 0.5 / ar
                                    calculate_data(dst_data, idx, center_x, center_y, box_width, box_height, False)

                if clip:
                    for i in (layer_height * layer_width * num_priors * 4):
                        dst_data[i] = min(max(dst_data[i], 0.0), 1.0)

                channel_size = out_shape[1]
                if len(variance) == 1:
                    for i in range(channel_size):
                        dst_data[i + channel_size] = variance[0]
                else:
                    for i in range(layer_height * layer_width * num_priors):
                        for j in range(4):
                            dst_data[i * 4 + j + channel_size] = variance[j]

                out = tf.constant(dst_data)
                inp = tf.reshape(out, shape=out_shape)

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### PriorBoxClustered - WIP
            elif layer.attrib['type'] == 'PriorBoxClustered':
                clip = False
                if not data is None and 'clip' in data.attrib:
                    clip = data.attrib['clip']
                    clip = True if clip.lower() == 'true' else False
                height = None
                if not data is None and 'height' in data.attrib:
                    height = np.asarray(data.attrib['height'].replace(' ', '').split(','), dtype=np.float32)
                width = None
                if not data is None and 'width' in data.attrib:
                    width = np.asarray(data.attrib['width'].replace(' ', '').split(','), dtype=np.float32)
                img_h = None
                if not data is None and 'img_h' in data.attrib:
                    img_h = np.asarray(data.attrib['img_h'], dtype=np.float32)
                img_w = None
                if not data is None and 'img_w' in data.attrib:
                    img_w = np.asarray(data.attrib['img_w'], dtype=np.float32)
                offset = None
                if not data is None and 'offset' in data.attrib:
                    offset = np.asarray(data.attrib['offset'], dtype=np.float32)
                step = None
                if not data is None and 'step' in data.attrib:
                    step = np.asarray(data.attrib['step'], dtype=np.float32)
                step_h = None
                if not data is None and 'step_h' in data.attrib:
                    step_h = np.asarray(data.attrib['step_h'], dtype=np.float32)
                step_w = None
                if not data is None and 'step_w' in data.attrib:
                    step_w = np.asarray(data.attrib['step_w'], dtype=np.float32)
                variance = None
                if not data is None and 'variance' in data.attrib:
                    variance = np.asarray(data.attrib['variance'].replace(' ', '').split(','), dtype=np.float32)

                port1 = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] # output_size
                port2 = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)] # image_size

                layer_width = port1[1]
                layer_height = port1[0]
                img_width = port2[1]
                img_height = port2[0]

                num_priors = len(width)
                if variance is None or len(variance) == 0:
                    variance = [0.1]
                var_size = len(variance)

                step_w = step if step_w == 0 else step_w
                step_h = step if step_h == 0 else step_h
                if (step_w == 0 and step_h == 0):
                    step_w = img_width / layer_width
                    step_h = img_height / layer_height

                out_shape = [2, 4 * layer_height * layer_width * num_priors]
                dst_data = np.zeros((out_shape), dtype=np.float32).flatten()

                for h in range(layer_height):
                    for w in range(layer_width):
                        center_x = (w + offset) * step_w
                        center_y = (h + offset) * step_h
                        for s in range(num_priors):
                            box_width = width[s]
                            box_height = height[s]
                            xmin = (center_x - box_width / 2.0) / img_width
                            ymin = (center_y - box_height / 2.0) / img_height
                            xmax = (center_x + box_width / 2.0) / img_width
                            ymax = (center_y + box_height / 2.0) / img_height
                            if clip:
                                xmin = min(max(xmin, 0.0), 1.0)
                                ymin = min(max(ymin, 0.0), 1.0)
                                xmax = min(max(xmax, 0.0), 1.0)
                                ymax = min(max(ymax, 0.0), 1.0)
                            def get_idx(cnt):
                                return h * layer_width * num_priors * cnt + w * num_priors * cnt + s * cnt
                            idx = get_idx(4)
                            dst_data[idx + 0] = xmin
                            dst_data[idx + 1] = ymin
                            dst_data[idx + 2] = xmax
                            dst_data[idx + 3] = ymax
                            idx = get_idx(var_size)
                            for j in range(var_size):
                                dst_data[idx + j + out_shape[1]] = variance[j]

                out = tf.constant(dst_data)
                inp = tf.reshape(out, shape=out_shape)

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                else:
                    tf_layers_dict[layer_id] = inp

            ### CumSum
            elif layer.attrib['type'] == 'CumSum':
                exclusive = False
                if not data is None and 'exclusive' in data.attrib:
                    exclusive = data.attrib['exclusive']
                    exclusive = True if exclusive.lower() == 'true' else False
                reverse = False
                if not data is None and 'reverse' in data.attrib:
                    reverse = data.attrib['reverse']
                    reverse = True if reverse.lower() == 'true' else False

                port1 = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                port2 = None
                try:
                    port2 = tf.cast(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)], dtype=tf.int32)[0]
                except:
                    port2 = tf.cast(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)], dtype=tf.int32)

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            port1
                        )
                        tf_layers_dict[layer_id] = tf.math.cumsum(
                            inp,
                            axis=port2,
                            exclusive=exclusive,
                            reverse=reverse
                        )

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.math.cumsum(
                            port1,
                            axis=port2,
                            exclusive=exclusive,
                            reverse=reverse
                        )
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )

                else:
                    tf_layers_dict[layer_id] = tf.math.cumsum(
                        port1,
                        axis=port2,
                        exclusive=exclusive,
                        reverse=reverse
                    )

            ### ReverseSequence
            elif layer.attrib['type'] == 'ReverseSequence':
                port1 = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                port2 = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]

                batch_axis = 0
                if not data is None and 'batch_axis' in data.attrib:
                    batch_axis = int(data.attrib['batch_axis'])
                seq_axis = [1]
                if not data is None and 'seq_axis' in data.attrib:
                    seq_axis = [int(data.attrib['seq_axis']) + 1]

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['type'] == 'ReverseSequence' and wr_config[layer_id]['replace_mode'] == 'change_batch_axis':
                        batch_axis = int(wr_config[layer_id]['values'])
                    elif wr_config[layer_id]['type'] == 'ReverseSequence' and wr_config[layer_id]['replace_mode'] == 'change_seq_axis':
                        seq_axis = int(wr_config[layer_id]['values'])

                if batch_axis > 0:
                    print(f'{Color.RED}ERROR:{Color.RESET} Only zero "batch_axis" is supported. layer_id: {layer_id}, batch_axis: {batch_axis}')
                    sys.exit(-1)

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            port1
                        )
                        tf_layers_dict[layer_id] = tf.reverse(
                            inp,
                            seq_axis
                        )
                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.reverse(
                            port1,
                            seq_axis
                        )
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                    else:
                        tf_layers_dict[layer_id] = tf.reverse(
                            port1,
                            seq_axis
                        )
                else:
                    tf_layers_dict[layer_id] = tf.reverse(
                        port1,
                        seq_axis
                    )

            ### ExtractImagePatches
            elif layer.attrib['type'] == 'ExtractImagePatches':
                port1 = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]

                auto_pad = None
                if not data is None and 'auto_pad' in data.attrib:
                    # same_upper, same_lower, valid - WIP Padding handling needs to be reviewed.
                    auto_pad = data.attrib['auto_pad'].upper()
                    if auto_pad == 'SAME_UPPER' or auto_pad == 'SAME_LOWER':
                        auto_pad = 'SAME'
                rates = None
                if not data is None and 'rates' in data.attrib:
                    rates = [int(val) for val in data.attrib['rates'].replace(' ', '').split(',')]
                    rates = [1, rates[0], rates[1], 1]
                sizes = None
                if not data is None and 'sizes' in data.attrib:
                    sizes = [int(val) for val in data.attrib['sizes'].replace(' ', '').split(',')]
                    sizes = [1, sizes[0], sizes[1], 1]
                strides = None
                if not data is None and 'strides' in data.attrib:
                    strides = [int(val) for val in data.attrib['strides'].replace(' ', '').split(',')]
                    strides = [1, strides[0], strides[1], 1]

                def pseudo_extract_image_patches(arr, ksizes, strides, rates, padding):
                    sizes = [1, ksizes[1]*rates[1] - (rates[1]-1), ksizes[2]*rates[2] - (rates[2]-1), 1]
                    if padding == 'SAME':
                        extra_i = max(0, (arr.shape[1]-1) // strides[1] * strides[1] + sizes[1] - arr.shape[1])
                        extra_j = max(0, (arr.shape[2]-1) // strides[2] * strides[2] + sizes[2] - arr.shape[2])
                        arr = np.pad(arr, [(0,0), (extra_i//2, extra_i//2 + extra_i%2), (extra_j//2, extra_j//2 + extra_j%2), (0,0)])
                    elif padding != 'VALID':
                        raise Exception('Padding type "%s" is not supported' % padding)
                    def make_range(in_size, k_size, rate, stride):
                        return range(0, in_size - (k_size*rate - rate), stride)
                    indexes_i = make_range(arr.shape[1], ksizes[1], rates[1], strides[1])
                    indexes_j = make_range(arr.shape[2], ksizes[2], rates[2], strides[2])
                    batch_size = arr.shape[0]
                    channel_size = ksizes[1]*ksizes[2]*arr.shape[3]
                    return tf.concat([tf.concat([tf.reshape(arr[:, i : sizes[1]+i : rates[1], j : sizes[2]+j : rates[2], :], [batch_size, 1, 1, channel_size]) for j in indexes_j], axis=2) for i in indexes_i], axis=1)

                image_patches_func = None
                # If the batch size is 1 and rates is [1,1], replace extract_image_patches with the standard operation.
                if port1.shape[0] == 1 and rates is not None and (rates[0] + rates[1]) == 2 and auto_pad == 'VALID':
                    image_patches_func = pseudo_extract_image_patches
                else:
                    image_patches_func = tf.image.extract_patches

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            port1
                        )
                        tf_layers_dict[layer_id] = image_patches_func(
                            inp,
                            ksizes=sizes,
                            strides=strides,
                            rates=rates,
                            padding=auto_pad
                        )

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = image_patches_func(
                            port1,
                            ksizes=sizes,
                            strides=strides,
                            rates=rates,
                            padding=auto_pad
                        )
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                    else:
                        tf_layers_dict[layer_id] = image_patches_func(
                            port1,
                            ksizes=sizes,
                            strides=strides,
                            rates=rates,
                            padding=auto_pad
                        )
                else:
                    tf_layers_dict[layer_id] = image_patches_func(
                        port1,
                        ksizes=sizes,
                        strides=strides,
                        rates=rates,
                        padding=auto_pad
                    )

            ### LogSoftmax
            elif layer.attrib['type'] == 'LogSoftmax':
                port1 = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]

                axis = None
                if not data is None and 'axis' in data.attrib:
                    axis = int(data.attrib['axis'])

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['type'] == 'LogSoftmax' and wr_config[layer_id]['replace_mode'] == 'change_axis':
                        axis = int(wr_config[layer_id]['values'])

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        inp = extrapolation_of_layers(
                            wr_config[layer_id],
                            port1
                        )
                        tf_layers_dict[layer_id] = \
                            (inp - tf.math.reduce_max(inp, axis)) - \
                                tf.math.log(tf.math.reduce_sum(tf.math.exp(inp - tf.math.reduce_max(inp, axis)), axis))

                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = \
                            (port1 - tf.math.reduce_max(port1, axis)) - \
                                tf.math.log(tf.math.reduce_sum(tf.math.exp(port1 - tf.math.reduce_max(port1, axis)), axis))
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                    else:
                        tf_layers_dict[layer_id] = \
                            (port1 - tf.math.reduce_max(port1, axis)) - \
                                tf.math.log(tf.math.reduce_sum(tf.math.exp(port1 - tf.math.reduce_max(port1, axis)), axis))
                else:
                    tf_layers_dict[layer_id] = \
                        (port1 - tf.math.reduce_max(port1, axis)) - \
                            tf.math.log(tf.math.reduce_sum(tf.math.exp(port1 - tf.math.reduce_max(port1, axis)), axis))

            ### Einsum
            elif layer.attrib['type'] == 'Einsum':
                ports = []
                port_idx = 0
                while True:
                    try:
                        ports.append(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, port_idx)])
                        port_idx += 1
                    except:
                        break
                equation = None
                if not data is None and 'equation' in data.attrib:
                    equation = data.attrib['equation']

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['type'] == 'Einsum' and wr_config[layer_id]['replace_mode'] == 'change_equation':
                        equation = wr_config[layer_id]['values']

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)
                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.einsum(equation, *ports)
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                    else:
                        tf_layers_dict[layer_id] = tf.einsum(equation, *ports)
                else:
                    tf_layers_dict[layer_id] =tf.einsum(equation, *ports)

            ### ScatterUpdate
            elif layer.attrib['type'] == 'ScatterUpdate':
                data = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                indices = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                updates = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]
                axis = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 3)]

                if isinstance(data, np.ndarray):
                    # data = tf.constant(data, dtype=cast_type_np_tf[str(data.dtype)])
                    data = tf.Variable(data, dtype=cast_type_np_tf[str(data.dtype)], trainable=False)
                if isinstance(indices, np.ndarray):
                    # indices = tf.constant(indices, dtype=cast_type_np_tf[str(indices.dtype)])
                    indices = tf.Variable(indices, dtype=cast_type_np_tf[str(indices.dtype)], trainable=False)
                if isinstance(updates, np.ndarray):
                    # updates = tf.constant(updates, dtype=cast_type_np_tf[str(updates.dtype)])
                    updates = tf.Variable(updates, dtype=cast_type_np_tf[str(updates.dtype)], trainable=False)

                print(f"{Color.YELLOW}WARNING:{Color.RESET} TensorFlow's ScatterUpdate ignores axis, which may result in incorrect processing results. layer_id: {layer_id}, axis: {axis}")

                if wr_config and layer_id in wr_config and format_version >= 2:
                    if wr_config[layer_id]['replace_mode'] == 'insert_before':
                        print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to {layer.attrib["type"]} {wr_config[layer_id]["replace_mode"]} is not supported. layer_id: {layer_id}')
                        sys.exit(-1)
                    elif wr_config[layer_id]['replace_mode'] == 'insert_after':
                        inp = tf.compat.v1.scatter_update(
                            data,
                            indices,
                            updates,
                        )
                        tf_layers_dict[layer_id] = extrapolation_of_layers(
                            wr_config[layer_id],
                            inp
                        )
                    else:
                        tf_layers_dict[layer_id] = tf.compat.v1.scatter_update(
                            data,
                            indices,
                            updates,
                        )
                else:
                    tf_layers_dict[layer_id] = tf.compat.v1.scatter_update(
                        data,
                        indices,
                        updates,
                    )

            ### Result
            elif layer.attrib['type'] == 'Result':
                tf_layers_dict[layer_id] = tf.identity(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    name=layer.attrib['name'].split('/')[0]
                )
                if layerids_of_the_terminating_output is None:
                    tf_outputs.append(tf_layers_dict[layer_id])

                if wr_config and layer_id in wr_config and format_version >= 2:
                    print(f'{Color.RED}ERROR:{Color.RESET} Extrapolation of operations to "Result" is not supported. layer_id: {layer_id}')
                    sys.exit(-1)

            else:
                print('The {} layer is not yet implemented.'.format(layer.attrib['type']))
                sys.exit(-1)


            # layerids_of_the_terminating_output
            if layerids_of_the_terminating_output is not None and layer_id in layerids_of_the_terminating_output:
                if layer.attrib['type'] != 'Split' and layer.attrib['type'] != 'VariadicSplit' and layer.attrib['type'] != 'TopK' and layer.attrib['type'] != 'NonMaxSuppression':
                    tf_outputs.append(tf_layers_dict[layer_id])
                else:
                    for layer_id_port in layer_id_port_dict[layer_id]['layer_id:port']:
                        tf_outputs.append(tf_layers_dict[layer_id_port])


            # Layer structure print
            if layer.attrib['type'] != 'Parameter' and layer.attrib['type'] != 'Const':
                try:
                    layer_structure = {
                        'layer_type': layer.attrib['type'],
                        'layer_id': layer_id,
                    }
                    for edge_index in range(len(tf_edges[layer_id])):
                        # if type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, edge_index)]) != np.ndarray:
                        if tf.keras.backend.is_keras_tensor(tf_layers_dict[layer_id]):
                            if not isinstance(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, edge_index)], np.ndarray):
                                layer_structure[f'input_layer{edge_index}'] = f'layer_id={tf_edges[layer_id][edge_index]}: {tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, edge_index)]}'
                            else:
                                if verbose:
                                    layer_structure[f'input_layer{edge_index}_value'] = f'layer_id={tf_edges[layer_id][edge_index]}: {tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, edge_index)]}'
                                else:
                                    layer_structure[f'input_layer{edge_index}_shape'] = f'layer_id={tf_edges[layer_id][edge_index]}: {tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, edge_index)].shape}'
                        else:
                            layer_structure[f'input_layer{edge_index}_shape'] = f'layer_id={tf_edges[layer_id][edge_index]}: Const(ndarray).shape {tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, edge_index)].shape}'
                            if verbose:
                                layer_structure[f'input_layer{edge_index}_value'] = f'layer_id={tf_edges[layer_id][edge_index]}: {tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, edge_index)]}'
                except:
                    pass

                if layer.attrib['type'] != 'Split' and layer.attrib['type'] != 'VariadicSplit' and layer.attrib['type'] != 'TopK' and layer.attrib['type'] != 'NonMaxSuppression':
                    if tf.keras.backend.is_keras_tensor(tf_layers_dict[layer_id]):
                        layer_structure['tf_layers_dict'] = tf_layers_dict[layer_id]
                    else:
                        layer_structure['tf_layers_dict_shape'] = tf_layers_dict[layer_id].shape
                        layer_structure['tf_layers_dict'] = tf_layers_dict[layer_id]

                    if layer.attrib['type'] == 'Concat' or layer.attrib['type'] == 'SoftMax' or layer.attrib['type'] == 'Squeeze' or \
                        layer.attrib['type'] == 'ReduceMean' or layer.attrib['type'] == 'ReduceMax' or layer.attrib['type'] == 'ReduceMin' or \
                            layer.attrib['type'] == 'ReduceSum' or  layer.attrib['type'] == 'ReduceProd' or layer.attrib['type'] == 'ReduceL2':
                        layer_structure['axis'] = axis

                    if layer.attrib['type'] == 'Unsqueeze':
                        layer_structure['indices'] = indices

                elif layer.attrib['type'] == 'Split' or layer.attrib['type'] == 'VariadicSplit' or layer.attrib['type'] == 'NonMaxSuppression':
                    # Split, VariadicSplit, NonMaxSuppression
                    for edge_index, tmp_layer_id_port in enumerate(layer_id_port_dict[layer_id]['layer_id:port']):
                        try:
                            if type(tf_layers_dict[get_tf_edges_from(tf_edges, tmp_layer_id_port, edge_index)]) != np.ndarray:
                                layer_structure[f'input_layer{edge_index}'] = f'layer_id={tf_edges[tmp_layer_id_port][edge_index]}: {tf_layers_dict[get_tf_edges_from(tf_edges, tmp_layer_id_port, edge_index)]}'
                            else:
                                layer_structure[f'input_layer{edge_index}'] = f'layer_id={tf_edges[tmp_layer_id_port][edge_index]}: Const(ndarray).shape {tf_layers_dict[get_tf_edges_from(tf_edges, tmp_layer_id_port, edge_index)].shape}'
                        except:
                            layer_structure[f'input_layer{edge_index}'] = f'layer_id=Unkown: Unkown'
                    for idx, (output, layer_id_port) in enumerate(zip(outputs, layer_id_port_dict[layer_id]['layer_id:port'])):
                        layer_structure[f'tf_layers_dict{idx}'] = f'layer_id_port: {layer_id_port} {output}'

                elif layer.attrib['type'] == 'TopK':
                    # TopK
                    layer_structure['tf_layers_dict0'] = tf_layers_dict[layer_id_values]
                    layer_structure['tf_layers_dict1'] = tf_layers_dict[layer_id_indices]

                layer_structure_print(layer_structure)

        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            print(f'{Color.RED}ERROR:{Color.RESET} model_path  : {model_path}.xml')
            print(f'{Color.RED}ERROR:{Color.RESET} weights_path: {model_path}.bin')
            print(f'{Color.RED}ERROR:{Color.RESET} layer_id    :', layer_id)
            try:
                for edge_index in range(len(tf_edges[layer_id])):
                    if type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, edge_index)]) != np.ndarray:
                        print(f'{Color.RED}ERROR:{Color.RESET} input_layer{edge_index} layer_id={tf_edges[layer_id][edge_index]}:', tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, edge_index)])
                    else:
                        print(f'{Color.RED}ERROR:{Color.RESET} input_layer{edge_index} layer_id={tf_edges[layer_id][edge_index]}: Const(ndarray).shape ', tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, edge_index)].shape)
                        pprint.pprint(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, edge_index)])
            except:
                pass
            print(f'{Color.RED}ERROR:{Color.RESET} The trace log is below.')
            import traceback
            traceback.print_exc()
            print(f'{Color.RED}ERROR:{Color.RESET} Please refer to 6-7 in the README first. https://github.com/PINTO0309/openvino2tensorflow')
            sys.exit(-1)

    # output If the layer type is ndarray, output it to a file as a numpy binary file and remove it from the model output
    np_outputs = [o for o in tf_outputs if isinstance(o, EagerTensor)]
    for idx, n in enumerate(np_outputs):
        np.save(f'{model_output_path}/{idx}', n)
        print(f'{Color.YELLOW}WARNING:{Color.RESET} The numpy array (ndarray) cannot be specified as an output layer. Therefore, the tool outputs a sequentially numbered .npy binary file. .npy_file_path: {model_output_path}/{idx}.npy')
    tf_outputs = [o for o in tf_outputs if not isinstance(o, EagerTensor)]

    model = Model(inputs=tf_inputs, outputs=tf_outputs)

    print(f'{Color.GREEN}TensorFlow/Keras model building process complete!{Color.RESET}')

    # saved_model output
    flag_for_output_switching_from_saved_model_to_pb_due_to_error = False
    if output_saved_model:
        try:
            print(f'{Color.REVERCE}saved_model output started{Color.RESET}', '=' * 58)
            tf.saved_model.save(model, model_output_path)
            # tf.keras.models.save_model(model, model_output_path, include_optimizer=False, save_format='tf', save_traces=False)
            # model.save(model_output_path, include_optimizer=False, save_format='tf', save_traces=False)
            print(f'{Color.GREEN}saved_model output complete!{Color.RESET}')
        except TypeError as e:
            print(f'{Color.GREEN}Switch to the output of an optimized protocol buffer file (.pb).{Color.RESET}')
            output_pb = True
            output_h5 = False
            flag_for_output_switching_from_saved_model_to_pb_due_to_error = True
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # .h5 output
    if output_h5:
        try:
            print(f'{Color.REVERCE}.h5 output started{Color.RESET}', '=' * 66)
            model.save(f'{model_output_path}/model_float32.h5', include_optimizer=False, save_format='h5')
            print(f'{Color.GREEN}.h5 output complete!{Color.RESET} - {model_output_path}/model_float32.h5')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # weight and json output
    if output_weight_and_json:
        try:
            print(f'{Color.REVERCE}weight and json output started{Color.RESET}', '=' * 54)
            open(f'{model_output_path}/model_float32.json', 'w').write(model.to_json())
            model.save_weights(f'{model_output_path}/model_float32_weights.h5')
            print(f'{Color.GREEN}weight and json output complete!{Color.RESET} - {model_output_path}/model_float32_weights.h5')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # .pb output
    if output_pb:
        try:
            print(f'{Color.REVERCE}.pb output started{Color.RESET}', '=' * 66)
            full_model = tf.function(lambda inputs: model(inputs))
            full_model = full_model.get_concrete_function(inputs=[tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in model.inputs])
            frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
            frozen_func.graph.as_graph_def()
            tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                                logdir=".",
                                name=f'{model_output_path}/model_float32.pb',
                                as_text=False)
            print(f'{Color.GREEN}.pb output complete!{Color.RESET} - {model_output_path}/model_float32.pb')

            if flag_for_output_switching_from_saved_model_to_pb_due_to_error:
                import shutil
                saved_model_tmp = 'saved_model_tmp'
                shutil.rmtree(saved_model_tmp, ignore_errors=True)
                os.makedirs(saved_model_tmp, exist_ok=True)
                inputs_tmp = []
                outputs_tmp = []
                for idx, _ in enumerate(model.inputs):
                    if idx == 0:
                        inputs_tmp.append(f'inputs:0')
                    else:
                        inputs_tmp.append(f'inputs_{idx}:0')
                for idx, _ in enumerate(model.outputs):
                    if idx == 0:
                        outputs_tmp.append(f'Identity:0')
                    else:
                        outputs_tmp.append(f'Identity_{idx}:0')

                def get_graph_def_from_file(graph_filepath):
                    tf.compat.v1.reset_default_graph()
                    tf.compat.v1.Graph().as_default()
                    with tf.compat.v1.gfile.GFile(graph_filepath, 'rb') as f:
                        graph_def = tf.compat.v1.GraphDef()
                        graph_def.ParseFromString(f.read())
                        return graph_def

                graph_def = get_graph_def_from_file(f'{model_output_path}/model_float32.pb')
                with tf.compat.v1.Session(graph=tf.Graph()) as sess:
                    tf.compat.v1.import_graph_def(graph_def, name='')
                    tf.compat.v1.saved_model.simple_save(
                        sess,
                        saved_model_tmp,
                        inputs= {t.rstrip(":0"):sess.graph.get_tensor_by_name(t) for t in inputs_tmp},
                        outputs={t.rstrip(":0"):sess.graph.get_tensor_by_name(t) for t in outputs_tmp}
                    )
                    from distutils.dir_util import copy_tree
                    copy_tree(saved_model_tmp, model_output_path)
                    shutil.rmtree(saved_model_tmp, ignore_errors=True)
                    print(f'{Color.GREEN}Optimized graph converted to SavedModel!{Color.RESET} - {model_output_path}')

        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # No Quantization - Input/Output=float32
    if output_no_quant_float32_tflite:
        try:
            print(f'{Color.REVERCE}tflite Float32 convertion started{Color.RESET}', '=' * 51)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            tflite_model = converter.convert()
            with open(f'{model_output_path}/model_float32.tflite', 'wb') as w:
                w.write(tflite_model)
            print(f'{Color.GREEN}tflite Float32 convertion complete!{Color.RESET} - {model_output_path}/model_float32.tflite')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # Dynamic Range Quantization - Input/Output=float32
    if output_dynamic_range_quant_tflite:
        try:
            print(f'{Color.REVERCE}Dynamic Range Quantization started{Color.RESET}', '=' * 50)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            tflite_model = converter.convert()
            with open(f'{model_output_path}/model_dynamic_range_quant.tflite', 'wb') as w:
                w.write(tflite_model)
            print(f'{Color.GREEN}Dynamic Range Quantization complete!{Color.RESET} - {model_output_path}/model_dynamic_range_quant.tflite')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # Weight Quantization - Input/Output=float32
    if output_weight_quant_tflite:
        try:
            print(f'{Color.REVERCE}Weight Quantization started{Color.RESET}', '=' * 57)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            tflite_model = converter.convert()
            with open(f'{model_output_path}/model_weight_quant.tflite', 'wb') as w:
                w.write(tflite_model)
            print(f'{Color.GREEN}Weight Quantization complete!{Color.RESET} - {model_output_path}/model_weight_quant.tflite')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # Float16 Quantization - Input/Output=float32
    if output_float16_quant_tflite:
        try:
            print(f'{Color.REVERCE}Float16 Quantization started{Color.RESET}', '=' * 56)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            tflite_quant_model = converter.convert()
            with open(f'{model_output_path}/model_float16_quant.tflite', 'wb') as w:
                w.write(tflite_quant_model)
            print(f'{Color.GREEN}Float16 Quantization complete!{Color.RESET} - {model_output_path}/model_float16_quant.tflite')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # Downloading datasets for calibration
    raw_test_data = None
    input_shapes = None
    if output_integer_quant_tflite or output_full_integer_quant_tflite:
        if calib_ds_type == 'tfds':
            print(f'{Color.REVERCE}TFDS download started{Color.RESET}', '=' * 63)
            raw_test_data = tfds.load(name=ds_name_for_tfds_for_calibration,
                                    with_info=False,
                                    split=split_name_for_tfds_for_calibration,
                                    data_dir=download_dest_folder_path_for_the_calib_tfds,
                                    download=tfds_download_flg)
            print(f'{Color.GREEN}TFDS download complete!{Color.RESET}')
        elif calib_ds_type == 'numpy':
            print(f'{Color.REVERCE}numpy dataset load started{Color.RESET}', '=' * 58)
            try:
                if load_dest_file_path_for_the_calib_npy == npy_load_default_path and not os.path.exists(npy_load_default_path):
                    os.makedirs(os.path.dirname(npy_load_default_path), exist_ok=True)
                    import gdown
                    import subprocess
                    try:
                        result = subprocess.check_output(
                            [
                                'gdown',
                                '--id', '1z-K0KZCK3JBH9hXFuBTmIM4jaMPOubGN',
                                '-O', load_dest_file_path_for_the_calib_npy
                            ],
                            stderr=subprocess.PIPE
                        ).decode('utf-8')
                    except:
                        result = subprocess.check_output(
                            [
                                'sudo', 'gdown',
                                '--id', '1z-K0KZCK3JBH9hXFuBTmIM4jaMPOubGN',
                                '-O', load_dest_file_path_for_the_calib_npy
                            ],
                            stderr=subprocess.PIPE
                        ).decode('utf-8')
                raw_test_data = np.load(load_dest_file_path_for_the_calib_npy)
                print(f'{Color.GREEN}numpy dataset load complete!{Color.RESET}')
            except subprocess.CalledProcessError as e:
                print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
                import traceback
                traceback.print_exc()
        else:
            pass
        input_shapes = [model_input.shape for model_input in model.inputs]

    def representative_dataset_gen():
        if calib_ds_type == 'tfds':
            for data in raw_test_data.take(10):
                image = data['image'].numpy()
                images = []
                for shape in input_shapes:
                    data = tf.image.resize(image, (shape[1], shape[2]))
                    tmp_image = eval(string_formulas_for_normalization) # Default: (data - [127.5,127.5,127.5]) / [127.5,127.5,127.5]
                    tmp_image = tmp_image[np.newaxis,:,:,:]
                    images.append(tmp_image)
                yield images
        elif calib_ds_type == 'numpy':
            for idx in range(raw_test_data.shape[0]):
                image = raw_test_data[idx]
                images = []
                for shape in input_shapes:
                    if len(shape) == 4 and shape[3] == 3:
                        data = tf.image.resize(image, (shape[1], shape[2]))
                        data = data[np.newaxis,:,:,:]
                    elif len(shape) == 4 and shape[3] == 1:
                        data = tf.image.resize(image, (shape[1], shape[2]))
                        data = 0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 2]
                        data = data[np.newaxis,:,:,np.newaxis]
                    else:
                        data = np.random.random_sample([i for i in shape]).astype(np.float32) * 255.0
                    tmp_image = eval(string_formulas_for_normalization) # Default: (data - [127.5,127.5,127.5]) / [127.5,127.5,127.5]
                    images.append(tmp_image)
                yield images

    # Integer Quantization
    if output_integer_quant_tflite:
        try:
            print(f'{Color.REVERCE}Integer Quantization started{Color.RESET}', '=' * 56)
            converter = tf.lite.TFLiteConverter.from_saved_model(model_output_path)
            converter.experimental_new_quantizer = use_experimental_new_quantizer
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
            converter.representative_dataset = representative_dataset_gen
            tflite_model = converter.convert()
            with open(f'{model_output_path}/model_integer_quant.tflite', 'wb') as w:
                w.write(tflite_model)
            print(f'{Color.GREEN}Integer Quantization complete!{Color.RESET} - {model_output_path}/model_integer_quant.tflite')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # Full Integer Quantization
    if output_full_integer_quant_tflite:
        try:
            print(f'{Color.REVERCE}Full Integer Quantization started{Color.RESET}', '=' * 51)
            converter = tf.lite.TFLiteConverter.from_saved_model(model_output_path)
            converter.experimental_new_quantizer = use_experimental_new_quantizer
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
            inf_type = None
            if output_integer_quant_type == 'int8':
                inf_type = tf.int8
            elif output_integer_quant_type == 'uint8':
                inf_type = tf.uint8
            else:
                inf_type = tf.int8
            converter.inference_input_type = inf_type
            converter.inference_output_type = inf_type
            converter.representative_dataset = representative_dataset_gen
            tflite_model = converter.convert()
            with open(f'{model_output_path}/model_full_integer_quant.tflite', 'wb') as w:
                w.write(tflite_model)
            print(f'{Color.GREEN}Full Integer Quantization complete!{Color.RESET} - {model_output_path}/model_full_integer_quant.tflite')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # TensorFlow.js convert
    if output_tfjs:
        import subprocess
        try:
            print(f'{Color.REVERCE}TensorFlow.js Float32 convertion started{Color.RESET}', '=' * 44)
            result = subprocess.check_output(
                [
                    'tensorflowjs_converter',
                    '--input_format', 'tf_saved_model',
                    '--output_format', 'tfjs_graph_model',
                    '--signature_name', 'serving_default',
                    '--saved_model_tags', 'serve',
                    model_output_path, f'{model_output_path}/tfjs_model_float32'
                ],
                stderr=subprocess.PIPE
            ).decode('utf-8')
            print(result)
            print(f'{Color.GREEN}TensorFlow.js convertion complete!{Color.RESET} - {model_output_path}/tfjs_model_float32')
        except subprocess.CalledProcessError as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
            import traceback
            traceback.print_exc()
        try:
            print(f'{Color.REVERCE}TensorFlow.js Float16 convertion started{Color.RESET}', '=' * 44)
            result = subprocess.check_output(
                [
                    'tensorflowjs_converter',
                    '--quantize_float16',
                    '--input_format', 'tf_saved_model',
                    '--output_format', 'tfjs_graph_model',
                    '--signature_name', 'serving_default',
                    '--saved_model_tags', 'serve',
                    model_output_path, f'{model_output_path}/tfjs_model_float16'
                ],
                stderr=subprocess.PIPE
            ).decode('utf-8')
            print(result)
            print(f'{Color.GREEN}TensorFlow.js convertion complete!{Color.RESET} - {model_output_path}/tfjs_model_float16')
        except subprocess.CalledProcessError as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
            import traceback
            traceback.print_exc()

    # TF-TRT (TensorRT) convert
    if output_tftrt_float32:
        try:
            def input_fn():
                input_shapes = []
                for tf_input in tf_inputs:
                    input_shapes.append(np.zeros(tf_input.shape).astype(np.float32))
                yield input_shapes

            print(f'{Color.REVERCE}TF-TRT (TensorRT) Float32 convertion started{Color.RESET}', '=' * 40)
            params = tf.experimental.tensorrt.ConversionParams(precision_mode='FP32', maximum_cached_engines=tftrt_maximum_cached_engines)
            converter = tf.experimental.tensorrt.Converter(input_saved_model_dir=model_output_path, conversion_params=params)
            converter.convert()
            converter.build(input_fn=input_fn)
            converter.save(f'{model_output_path}/tensorrt_saved_model_float32')
            print(f'{Color.GREEN}TF-TRT (TensorRT) convertion complete!{Color.RESET} - {model_output_path}/tensorrt_saved_model_float32')

        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()
            print(f'{Color.RED}The binary versions of TensorFlow and TensorRT may not be compatible. Please check the version compatibility of each package.{Color.RESET}')
            print(f'{Color.RED}--tftrt_maximum_cached_engines should be less than 10000 and try to convert again. It is very likely that the GPU memory has been exhausted.{Color.RESET}')
            print(f'{Color.RED}Try using the nvidia-smi command during the conversion process to see how much GPU RAM is consumed.{Color.RESET}')

    if output_tftrt_float16:
        try:
            def input_fn():
                input_shapes = []
                for tf_input in tf_inputs:
                    input_shapes.append(np.zeros(tf_input.shape).astype(np.float32))
                yield input_shapes

            print(f'{Color.REVERCE}TF-TRT (TensorRT) Float16 convertion started{Color.RESET}', '=' * 40)
            params = tf.experimental.tensorrt.ConversionParams(precision_mode='FP16', maximum_cached_engines=tftrt_maximum_cached_engines)
            converter = tf.experimental.tensorrt.Converter(input_saved_model_dir=model_output_path, conversion_params=params)
            converter.convert()
            converter.build(input_fn=input_fn)
            converter.save(f'{model_output_path}/tensorrt_saved_model_float16')
            print(f'{Color.GREEN}TF-TRT (TensorRT) convertion complete!{Color.RESET} - {model_output_path}/tensorrt_saved_model_float16')

        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()
            print(f'{Color.RED}The binary versions of TensorFlow and TensorRT may not be compatible. Please check the version compatibility of each package.{Color.RESET}')
            print(f'{Color.RED}--tftrt_maximum_cached_engines should be less than 10000 and try to convert again. It is very likely that the GPU memory has been exhausted.{Color.RESET}')
            print(f'{Color.RED}Try using the nvidia-smi command during the conversion process to see how much GPU RAM is consumed.{Color.RESET}')

    # CoreML convert
    if output_coreml:
        try:
            print(f'{Color.REVERCE}CoreML convertion started{Color.RESET}', '=' * 59)
            mlmodel = ct.convert(model_output_path, source='tensorflow')
            mlmodel.save(f'{model_output_path}/model_coreml_float32.mlmodel')
            print(f'{Color.GREEN}CoreML convertion complete!{Color.RESET} - {model_output_path}/model_coreml_float32.mlmodel')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # EdgeTPU convert
    if output_edgetpu:
        try:
            print(f'{Color.REVERCE}EdgeTPU convertion started{Color.RESET}', '=' * 58)
            result = subprocess.check_output(
                [
                    'edgetpu_compiler',
                    '-o', model_output_path,
                    '-sad',
                    '-t', str(edgetpu_compiler_timeout),
                    '-n', str(edgetpu_num_segments),
                    f'{model_output_path}/model_full_integer_quant.tflite'
                ],
                stderr=subprocess.PIPE
            ).decode('utf-8')
            print(result)
            print(f'{Color.GREEN}EdgeTPU convert complete!{Color.RESET} - {model_output_path}/model_full_integer_quant_edgetpu.tflite')
        except subprocess.CalledProcessError as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
            import traceback
            traceback.print_exc()
            print("-" * 80)
            print('Please install edgetpu_compiler according to the following website.')
            print('https://coral.ai/docs/edgetpu/compiler/#system-requirements')

    # ONNX convert
    if output_onnx:
        import onnx
        import onnxoptimizer
        import subprocess
        try:
            print(f'{Color.REVERCE}ONNX convertion started{Color.RESET}', '=' * 61)
            loaded = tf.saved_model.load(model_output_path).signatures['serving_default']
            inputs = ",".join(map(str, [inp.name for inp in loaded.inputs if 'unknown' not in inp.name])).rstrip(',')
            onnx_convert_command = None
            if not onnx_extra_opset:
                onnx_convert_command = \
                [
                    'python3',
                    '-m', 'tf2onnx.convert',
                    '--saved-model', model_output_path,
                    '--opset', str(onnx_opset),
                    '--output', f'{model_output_path}/model_float32.onnx'
                ]
                if use_onnx_nchw_conversion:
                    onnx_convert_command.append(
                        '--inputs-as-nchw'
                    )
                    onnx_convert_command.append(
                        f'{inputs}'
                    )
            else:
                onnx_convert_command = \
                [
                    'python3',
                    '-m', 'tf2onnx.convert',
                    '--saved-model', model_output_path,
                    '--opset', str(onnx_opset),
                    '--output', f'{model_output_path}/model_float32.onnx',
                    '--extra_opset', onnx_extra_opset
                ]
                if use_onnx_nchw_conversion:
                    onnx_convert_command.append(
                        '--inputs-as-nchw'
                    )
                    onnx_convert_command.append(
                        f'{inputs}'
                    )
            result = subprocess.check_output(
                onnx_convert_command,
                stderr=subprocess.PIPE
            ).decode('utf-8')
            try:
                onnx_model = onnx.load(f'{model_output_path}/model_float32.onnx')
                onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
                onnx.save(onnx_model, f'{model_output_path}/model_float32.onnx')
            except Exception as e:
                print(f'{Color.YELLOW}WARNING:{Color.RESET}', e)
                import traceback
                traceback.print_exc()
            print(result)
            print(f'{Color.GREEN}ONNX convertion complete!{Color.RESET} - {model_output_path}/model_float32.onnx')
        except subprocess.CalledProcessError as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
            import traceback
            traceback.print_exc()

        if use_onnx_optimization:
            try:
                print(f'{Color.REVERCE}ONNX optimization started{Color.RESET}', '=' * 59)

                # onnxoptimizer
                onnx_model = onnx.load(f'{model_output_path}/model_float32.onnx')
                passes = [
                    "extract_constant_to_initializer",
                    "eliminate_unused_initializer"
                ]
                optimized_model = onnxoptimizer.optimize(onnx_model, passes)
                onnx.save(optimized_model, f'{model_output_path}/model_float32.onnx')

                # onnx-simplifier
                result = subprocess.check_output(
                    [
                        'python3',
                        '-m', 'onnxsim',
                        f'{model_output_path}/model_float32.onnx',
                        f'{model_output_path}/model_float32.onnx'
                    ],
                    stderr=subprocess.PIPE
                ).decode('utf-8')
                print(result)

                print(f'{Color.GREEN}ONNX optimization complete!{Color.RESET} - {model_output_path}/model_float32.onnx')
            except subprocess.CalledProcessError as e:
                print(f'{Color.YELLOW}WARNING:{Color.RESET}', e.stderr.decode('utf-8'))
                import traceback
                traceback.print_exc()

    # Myriad Inference Engine blob
    if output_myriad:
        try:
            print(f'{Color.REVERCE}Myriad Inference Engine blob convertion started{Color.RESET}', '=' * 44)
            os.makedirs(f'{model_output_path}/openvino/myriad', exist_ok=True)
            INTEL_OPENVINO_DIR = os.environ['INTEL_OPENVINO_DIR']
            result = subprocess.check_output(
                [
                    f'{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/myriad_compile',
                    '-m', f'{model_path}.xml',
                    '-VPU_NUMBER_OF_SHAVES', f'{vpu_number_of_shaves}',
                    '-VPU_NUMBER_OF_CMX_SLICES', f'{vpu_number_of_cmx_slices}',
                    '-o', f'{model_output_path}/openvino/myriad/saved_model.blob'
                ],
                stderr=subprocess.PIPE
            ).decode('utf-8')
            print(result)
            print(f'{Color.GREEN}Myriad Inference Engine blob convertion complete!{Color.RESET} - {model_output_path}/openvino/myriad')
        except subprocess.CalledProcessError as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='input IR model path (.xml)')
    parser.add_argument('--model_output_path', type=str, default='saved_model', help='The output folder path of the converted model file')
    parser.add_argument('--output_saved_model', action='store_true', help='saved_model output switch')
    parser.add_argument('--output_h5', action='store_true', help='.h5 output switch')
    parser.add_argument('--output_weight_and_json', action='store_true', help='weight of h5 and json output switch')
    parser.add_argument('--output_pb', action='store_true', help='.pb output switch')
    parser.add_argument('--output_no_quant_float32_tflite', action='store_true', help='float32 tflite output switch')
    parser.add_argument('--output_dynamic_range_quant_tflite', action='store_true', help='dynamic range quant tflite output switch')
    parser.add_argument('--output_weight_quant_tflite', action='store_true', help='weight quant tflite output switch')
    parser.add_argument('--output_float16_quant_tflite', action='store_true', help='float16 quant tflite output switch')
    parser.add_argument('--output_integer_quant_tflite', action='store_true', help='integer quant tflite output switch')
    parser.add_argument('--output_full_integer_quant_tflite', action='store_true', help='full integer quant tflite output switch')
    parser.add_argument('--output_integer_quant_type', type=str, default='int8', help='Input and output types when doing Integer Quantization (\'int8 (default)\' or \'uint8\')')
    parser.add_argument('--string_formulas_for_normalization', type=str, default='(data - [127.5,127.5,127.5]) / [127.5,127.5,127.5]', help='String formulas for normalization. It is evaluated by Python\'s eval() function. Default: \'(data - [127.5,127.5,127.5]) / [127.5,127.5,127.5]\'')
    parser.add_argument('--calib_ds_type', type=str, default='numpy', help='Types of data sets for calibration. tfds or numpy. Only one of them can be specified. Default: numpy [20, 513, 513, 3] -> [Number of images, h, w, c]')
    parser.add_argument('--ds_name_for_tfds_for_calibration', type=str, default='coco/2017', help='Dataset name for TensorFlow Datasets for calibration. https://www.tensorflow.org/datasets/catalog/overview')
    parser.add_argument('--split_name_for_tfds_for_calibration', type=str, default='validation', help='Split name for TensorFlow Datasets for calibration. https://www.tensorflow.org/datasets/catalog/overview')
    tfds_dl_default_path = f'{str(Path.home())}/TFDS'
    parser.add_argument('--download_dest_folder_path_for_the_calib_tfds', type=str, default=tfds_dl_default_path, help='Download destination folder path for the calibration dataset. Default: $HOME/TFDS')
    parser.add_argument('--tfds_download_flg', action='store_true', help='True to automatically download datasets from TensorFlow Datasets. True or False')
    npy_load_default_path = 'sample_npy/calibration_data_img_sample.npy'
    parser.add_argument('--load_dest_file_path_for_the_calib_npy', type=str, default=npy_load_default_path, help='The path from which to load the .npy file containing the numpy binary version of the calibration data. Default: sample_npy/calibration_data_img_sample.npy')
    parser.add_argument('--output_tfjs', action='store_true', help='tfjs model output switch')
    parser.add_argument('--output_tftrt_float32', action='store_true', help='tftrt float32 model output switch')
    parser.add_argument('--output_tftrt_float16', action='store_true', help='tftrt float16 model output switch')
    parser.add_argument('--tftrt_maximum_cached_engines', type=int, default=10000, help='Specifies the quantity of tftrt_maximum_cached_engines for TFTRT. Default: 10000')
    parser.add_argument('--output_coreml', action='store_true', help='coreml model output switch')
    parser.add_argument('--output_edgetpu', action='store_true', help='edgetpu model output switch')
    parser.add_argument('--edgetpu_compiler_timeout', type=int, default=3600, help='edgetpu_compiler timeout for one compilation process in seconds. Default: 3600')
    parser.add_argument('--edgetpu_num_segments', type=int, default=1, help='Partition the model into [num_segments] segments. Default: 1 (no partition)')
    parser.add_argument('--output_onnx', action='store_true', help='onnx model output switch')
    parser.add_argument('--onnx_opset', type=int, default=13, help='onnx opset version number')
    parser.add_argument('--onnx_extra_opset', type=str, default='', help='The name of the onnx extra_opset to enable. Default: \'\'. "com.microsoft:1" or "ai.onnx.contrib:1" or "ai.onnx.ml:1"')
    parser.add_argument('--disable_onnx_nchw_conversion', action='store_true', help='Disable onnx NCHW conversion.')
    parser.add_argument('--disable_onnx_optimization', action='store_true', help='Disable onnx optimization')
    parser.add_argument('--output_myriad', action='store_true', help='myriad inference engine blob output switch')
    parser.add_argument('--vpu_number_of_shaves', type=int, default=4, help='vpu number of shaves. Default: 4')
    parser.add_argument('--vpu_number_of_cmx_slices', type=int, default=4, help='vpu number of cmx slices. Default: 4')
    parser.add_argument('--replace_swish_and_hardswish', action='store_true', help='Replace swish and hard-swish with each other')
    parser.add_argument('--optimizing_hardswish_for_edgetpu', action='store_true', help='Optimizing hardswish for edgetpu')
    parser.add_argument('--replace_prelu_and_minmax', action='store_true', help='Replace prelu and minimum/maximum with each other')
    parser.add_argument('--replace_argmax', action='store_true', help='Replace ArgMax with a primitive operation')
    parser.add_argument('--replace_argmax_indices_to_float32', action='store_true', help='Enabling this option may allow full mapping to EdgeTPU when ArgMax is at the end of the model for tasks such as SemanticSegmentation. If you apply it to ArgMax, which is located in the middle of the model, the model transformation is more likely to fail.')
    parser.add_argument('--restricted_resize_image_mode', action='store_true', help='Specify this if the upsampling contains OPs that are not scaled by integer multiples. Optimization for EdgeTPU will be disabled.')
    parser.add_argument('--weight_replacement_config', type=str, default='', help='Replaces the value of Const for each layer_id defined in json. Specify the path to the json file. "weight_replacement_config.json"')
    parser.add_argument('--disable_experimental_new_quantizer', action='store_true', help='Disable MLIR\'s new quantization feature during INT8 quantization in TensorFlowLite.')
    parser.add_argument('--optimizing_barracuda', action='store_true', help='Generates ONNX by replacing Barracuda\'s unsupported layers with standard layers.')
    parser.add_argument('--layerids_of_the_terminating_output', type=str, default='', help='A comma-separated list of layer IDs to be used as output layers. Default: \'\'')
    parser.add_argument('--keep_input_tensor_in_nchw', action='store_true', help='Does not convert the input to NHWC, but keeps the NCHW format. Transpose is inserted right after the input layer, and the model internals are handled by NHWC. Only 4D input is supported.')
    parser.add_argument('--non_verbose', action='store_true', help='Do not show all the weight information of each layer in the conversion log')
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
    output_dynamic_range_quant_tflite = args.output_dynamic_range_quant_tflite
    output_weight_quant_tflite = args.output_weight_quant_tflite
    output_float16_quant_tflite = args.output_float16_quant_tflite
    output_integer_quant_tflite = args.output_integer_quant_tflite
    output_full_integer_quant_tflite = args.output_full_integer_quant_tflite
    output_integer_quant_type = args.output_integer_quant_type.lower()
    string_formulas_for_normalization = args.string_formulas_for_normalization.lower()
    calib_ds_type = args.calib_ds_type.lower()
    ds_name_for_tfds_for_calibration = args.ds_name_for_tfds_for_calibration
    split_name_for_tfds_for_calibration = args.split_name_for_tfds_for_calibration
    download_dest_folder_path_for_the_calib_tfds = args.download_dest_folder_path_for_the_calib_tfds
    tfds_download_flg = args.tfds_download_flg
    load_dest_file_path_for_the_calib_npy = args.load_dest_file_path_for_the_calib_npy
    output_tfjs = args.output_tfjs
    output_tftrt_float32 = args.output_tftrt_float32
    output_tftrt_float16 = args.output_tftrt_float16
    tftrt_maximum_cached_engines = args.tftrt_maximum_cached_engines
    output_coreml = args.output_coreml
    output_edgetpu = args.output_edgetpu
    edgetpu_compiler_timeout = args.edgetpu_compiler_timeout
    edgetpu_num_segments = args.edgetpu_num_segments
    output_onnx = args.output_onnx
    onnx_opset = args.onnx_opset
    onnx_extra_opset = args.onnx_extra_opset
    use_onnx_nchw_conversion = not args.disable_onnx_nchw_conversion
    use_onnx_optimization = not args.disable_onnx_optimization
    output_myriad = args.output_myriad
    vpu_number_of_shaves = args.vpu_number_of_shaves
    vpu_number_of_cmx_slices = args.vpu_number_of_cmx_slices
    replace_swish_and_hardswish = args.replace_swish_and_hardswish
    optimizing_hardswish_for_edgetpu = args.optimizing_hardswish_for_edgetpu
    replace_prelu_and_minmax = args.replace_prelu_and_minmax
    replace_argmax = args.replace_argmax
    replace_argmax_indices_to_float32 = args.replace_argmax_indices_to_float32
    restricted_resize_image_mode = args.restricted_resize_image_mode
    weight_replacement_config = args.weight_replacement_config
    use_experimental_new_quantizer = not args.disable_experimental_new_quantizer
    optimizing_barracuda = args.optimizing_barracuda
    layerids_of_the_terminating_output_tmp = args.layerids_of_the_terminating_output
    layerids_of_the_terminating_output = None
    if layerids_of_the_terminating_output_tmp:
        layerids_of_the_terminating_output = [ids.strip() for ids in layerids_of_the_terminating_output_tmp.split(',')]
    keep_input_tensor_in_nchw = args.keep_input_tensor_in_nchw
    verbose = not args.non_verbose

    if not output_saved_model and \
        not output_h5 and \
        not output_dynamic_range_quant_tflite and \
        not output_weight_and_json and \
        not output_pb and \
        not output_no_quant_float32_tflite and \
        not output_weight_quant_tflite and \
        not output_float16_quant_tflite and \
        not output_integer_quant_tflite and \
        not output_full_integer_quant_tflite and \
        not output_tfjs and \
        not output_tftrt_float32 and \
        not output_tftrt_float16 and \
        not output_coreml and \
        not output_edgetpu and \
        not output_onnx and \
        not output_myriad:
        print('Set at least one of the output switches (output_*) to true.')
        sys.exit(-1)

    if output_edgetpu:
        output_full_integer_quant_tflite = True
        replace_argmax = True

    from pkg_resources import working_set
    package_list = []
    for dist in working_set:
        package_list.append(dist.project_name)

    if output_tfjs:
        if not 'tensorflowjs' in package_list:
            print('\'tensorflowjs\' is not installed. Please run the following command to install \'tensorflowjs\'.')
            print('pip3 install --upgrade tensorflowjs')
            sys.exit(-1)
    if output_tftrt_float32 or output_tftrt_float16:
        if not 'tensorrt' in package_list:
            print('\'tensorrt\' is not installed. Please check the following website and install \'tensorrt\'.')
            print('https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html')
            sys.exit(-1)
    if output_coreml:
        if not 'coremltools' in package_list:
            print('\'coremltoos\' is not installed. Please run the following command to install \'coremltoos\'.')
            print('pip3 install --upgrade coremltools')
            sys.exit(-1)
    if output_onnx:
        if not 'tf2onnx' in package_list:
            print('\'tf2onnx\' is not installed. Please run the following command to install \'tf2onnx\'.')
            print('pip3 install --upgrade onnx')
            print('pip3 install --upgrade tf2onnx')
            sys.exit(-1)

    if output_integer_quant_tflite or output_full_integer_quant_tflite:
        if not 'tensorflow-datasets' in package_list:
            print('\'tensorflow-datasets\' is not installed. Please run the following command to install \'tensorflow-datasets\'.')
            print('pip3 install --upgrade tensorflow-datasets')
            sys.exit(-1)

    if output_coreml and output_edgetpu:
        print(f'{Color.RED}ERROR:{Color.RESET} output_coreml and output_edgetpu cannot be True at the same time.')
        sys.exit(-1)

    if output_integer_quant_type == 'int8' or output_integer_quant_type == 'uint8':
        pass
    else:
        print('Only \'int8\' or \'uint8\' can be specified for output_integer_quant_type.')
        sys.exit(-1)

    if calib_ds_type == 'tfds':
        pass
    elif calib_ds_type == 'numpy':
        pass
    else:
        print('Only \'tfds\' or \'numpy\' can be specified for calib_ds_type.')
        sys.exit(-1)

    if weight_replacement_config and not os.path.exists(weight_replacement_config):
        print('The json file does not exist in the path specified in weight_replacement_config.')
        sys.exit(-1)

    if layerids_of_the_terminating_output:
        for s in layerids_of_the_terminating_output:
            if not s.isdecimal():
                print('layerids_of_the_terminating_output should be specified as a comma-separated integer value.')
                sys.exit(-1)

    del package_list
    os.makedirs(model_output_path, exist_ok=True)
    convert(model, model_output_path, output_saved_model, output_h5, output_weight_and_json, output_pb,
            output_no_quant_float32_tflite, output_dynamic_range_quant_tflite, output_weight_quant_tflite, output_float16_quant_tflite,
            output_integer_quant_tflite, output_full_integer_quant_tflite, output_integer_quant_type,
            string_formulas_for_normalization,
            calib_ds_type, ds_name_for_tfds_for_calibration, split_name_for_tfds_for_calibration,
            download_dest_folder_path_for_the_calib_tfds, tfds_download_flg, npy_load_default_path, load_dest_file_path_for_the_calib_npy,
            output_tfjs, output_tftrt_float32, output_tftrt_float16, tftrt_maximum_cached_engines, output_coreml,
            output_edgetpu, edgetpu_compiler_timeout, edgetpu_num_segments,
            output_onnx, onnx_opset, onnx_extra_opset, use_onnx_nchw_conversion, use_onnx_optimization, output_myriad,
            vpu_number_of_shaves, vpu_number_of_cmx_slices,
            replace_swish_and_hardswish, optimizing_hardswish_for_edgetpu, replace_prelu_and_minmax, replace_argmax, replace_argmax_indices_to_float32,
            restricted_resize_image_mode, weight_replacement_config, use_experimental_new_quantizer,
            optimizing_barracuda, layerids_of_the_terminating_output, keep_input_tensor_in_nchw, verbose)
    print(f'{Color.REVERCE}All the conversion process is finished!{Color.RESET}', '=' * 45)

if __name__ == "__main__":
    main()