# openvino2tensorflow

<p align="center">
  <img src="https://user-images.githubusercontent.com/33194443/104584047-4e688f80-56a5-11eb-8dc2-5816487239d0.png" />
</p>

This script converts the OpenVINO IR model to Tensorflow's saved_model, tflite, h5, tfjs, tftrt(TensorRT), CoreML, EdgeTPU, ONNX and pb. PyTorch (NCHW) -> ONNX (NCHW) -> OpenVINO (NCHW) -> openvino2tensorflow -> Tensorflow/Keras (NHWC) -> TFLite (NHWC). And the conversion from .pb to saved_model and from saved_model to .pb and from .pb to .tflite and saved_model to .tflite and saved_model to onnx. Support for building environments with Docker. It is possible to directly access the host PC GUI and the camera to verify the operation. NVIDIA GPU (dGPU) support. Intel iHD GPU (iGPU) support.

[Special custom TensorFlow binaries](https://github.com/PINTO0309/Tensorflow-bin) and [special custom TensorFLow Lite binaries](https://github.com/PINTO0309/TensorflowLite-bin) are used.

Work in progress now.

**I'm continuing to add more layers of support and bug fixes on a daily basis. If you have a model that you are having trouble converting, please share the `.bin` and `.xml` with the issue. I will try to convert as much as possible.**

[![PyPI - Downloads](https://img.shields.io/pypi/dm/openvino2tensorflow?color=2BAF2B&label=Downloads%EF%BC%8FInstalled)](https://pypistats.org/packages/openvino2tensorflow) ![GitHub](https://img.shields.io/github/license/PINTO0309/openvino2tensorflow?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/openvino2tensorflow?color=2BAF2B)](https://pypi.org/project/openvino2tensorflow/)

![ezgif com-gif-maker (4)](https://user-images.githubusercontent.com/33194443/103457894-4ffd9380-4d46-11eb-86dd-f753f2fca093.gif)

![ezgif com-gif-maker (3)](https://user-images.githubusercontent.com/33194443/103456348-8d5b2480-4d38-11eb-8a58-9b7c7203b18c.gif)

## 1. Environment
- TensorFlow v2.6.0+
- OpenVINO 2021.4.582+
- Python 3.6+
- tensorflowjs **`pip3 install --upgrade tensorflowjs`**
- **[tensorrt](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)**
- coremltools **`pip3 install --upgrade coremltools`**
- onnx **`pip3 install --upgrade onnx`**
- tf2onnx **`pip3 install --upgrade tf2onnx`**
- tensorflow-datasets **`pip3 install --upgrade tensorflow-datasets`**
- **[edgetpu_compiler](https://coral.ai/docs/edgetpu/compiler/#system-requirements)**
- Intel-Media-SDK
- Intel iHD GPU (iGPU) support
- Docker

## 2. Use case

- PyTorch (NCHW) -> ONNX (NCHW) -> OpenVINO (NCHW) ->
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TFLite (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TFJS (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TF-TRT (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> EdgeTPU (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> CoreML (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> ONNX (NHWC)
  - -> **`openvino2tensorflow`** -> Myriad Inference Engine Blob (NCHW)

- Caffe (NCHW) -> OpenVINO (NCHW) ->
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TFLite (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TFJS (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TF-TRT (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> EdgeTPU (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> CoreML (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> ONNX (NHWC)
  - -> **`openvino2tensorflow`** -> Myriad Inference Engine Blob (NCHW)

- MXNet (NCHW) -> OpenVINO (NCHW) ->
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TFLite (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TFJS (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TF-TRT (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> EdgeTPU (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> CoreML (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> ONNX (NHWC)
  - -> **`openvino2tensorflow`** -> Myriad Inference Engine Blob (NCHW)

- Keras (NHWC) -> OpenVINO (NCHW・Optimized) ->
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TFLite (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TFJS (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TF-TRT (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> EdgeTPU (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> CoreML (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> ONNX (NHWC)
  - -> **`openvino2tensorflow`** -> Myriad Inference Engine Blob (NCHW)

- saved_model -> **`saved_model_to_pb`** -> pb

- saved_model ->
  - -> **`saved_model_to_tflite`** -> TFLite
  - -> **`saved_model_to_tflite`** -> TFJS
  - -> **`saved_model_to_tflite`** -> TF-TRT
  - -> **`saved_model_to_tflite`** -> EdgeTPU
  - -> **`saved_model_to_tflite`** -> CoreML
  - -> **`saved_model_to_tflite`** -> ONNX

- pb -> **`pb_to_tflite`** -> TFLite

- pb -> **`pb_to_saved_model`** -> saved_model

## 3. Supported Layers
- Currently, there are problems with the **`Reshape`** and **`Transpose`** operation of 5D Tensor. Since it is difficult to accurately predict the shape of a simple shape change, I have added support for forced replacement of transposition parameters using JSON files. [#6-7-replace-weights-or-constant-values-in-const-op](#6-7-replace-weights-or-constant-values-in-const-op)

|No.|OpenVINO Layer|TF Layer|Remarks|
|:--:|:--|:--|:--|
|1|Parameter|Input||
|2|Const|Constant, Bias||
|3|Convolution|Conv2D||
|4|Add|Add||
|5|ReLU|ReLU||
|6|PReLU|PReLU|Maximum(0.0,x)+alpha\*Minimum(0.0,x)|
|7|MaxPool|MaxPool2D||
|8|AvgPool|AveragePooling2D||
|9|GroupConvolution|DepthwiseConv2D, Conv2D/Split/Concat||
|10|ConvolutionBackpropData|Conv2DTranspose||
|11|Concat|Concat||
|12|Multiply|Multiply||
|13|Tan|Tan||
|14|Tanh|Tanh||
|15|Elu|Elu||
|16|Sigmoid|Sigmoid||
|17|HardSigmoid|hard_sigmoid||
|18|SoftPlus|SoftPlus||
|19|Swish|Swish|You can replace swish and hard-swish with each other by using the "--replace_swish_and_hardswish" option|
|20|Interpolate|ResizeNearestNeighbor, ResizeBilinear||
|21|ShapeOf|Shape||
|22|Convert|Cast||
|23|StridedSlice|Strided_Slice||
|24|Pad|Pad, MirrorPad||
|25|Clamp|ReLU6, Clip||
|26|TopK|ArgMax, top_k||
|27|Transpose|Transpose||
|28|Squeeze|Squeeze||
|29|Unsqueeze|Identity, expand_dims|WIP|
|30|ReduceMean|reduce_mean||
|31|ReduceMax|reduce_max||
|32|ReduceMin|reduce_min||
|33|ReduceSum|reduce_sum||
|34|ReduceProd|reduce_prod||
|35|Subtract|Subtract||
|36|MatMul|MatMul||
|37|Reshape|Reshape||
|38|Range|Range|WIP|
|39|Exp|Exp||
|40|Abs|Abs||
|41|SoftMax|SoftMax||
|42|Negative|Negative||
|43|Maximum|Maximum|No broadcast|
|44|Minimum|Minimum|No broadcast|
|45|Acos|Acos||
|46|Acosh|Acosh||
|47|Asin|Asin||
|48|Asinh|Asinh||
|49|Atan|Atan||
|50|Atanh|Atanh||
|51|Ceiling|Ceil||
|52|Cos|Cos||
|53|Cosh|Cosh||
|54|Sin|Sin||
|55|Sinh|Sinh||
|56|Gather|Gather||
|57|Divide|Divide, FloorDiv||
|58|Erf|Erf||
|59|Floor|Floor||
|60|FloorMod|FloorMod||
|61|HSwish|HardSwish|x\*ReLU6(x+3)\*0.16666667, You can replace swish and hard-swish with each other by using the "--replace_swish_and_hardswish" option|
|62|Log|Log||
|63|Power|Pow|No broadcast|
|64|Mish|Mish|x\*Tanh(softplus(x))|
|65|Selu|Selu||
|66|Equal|equal||
|67|NotEqual|not_equal||
|68|Greater|greater||
|69|GreaterEqual|greater_equal||
|70|Less|less||
|71|LessEqual|less_equal||
|72|Select|Select|No broadcast|
|73|LogicalAnd|logical_and||
|74|LogicalNot|logical_not||
|75|LogicalOr|logical_or||
|76|LogicalXor|logical_xor||
|77|Broadcast|broadcast_to, ones, Multiply|numpy / bidirectional mode, WIP|
|78|Split|Split||
|79|VariadicSplit|Split, Slice, SplitV||
|80|MVN|reduce_mean, sqrt, reduce_variance|(x-reduce_mean(x))/sqrt(reduce_variance(x)+eps)|
|81|NonZero|not_equal, boolean_mask||
|82|ReduceL2|Multiply, reduce_sum, rsqrt||
|83|SpaceToDepth|SpaceToDepth||
|84|DepthToSpace|DepthToSpace||
|85|Sqrt|sqrt||
|86|SquaredDifference|squared_difference||
|87|FakeQuantize|subtract, multiply, round, greater, where, less_equal, add||
|88|Tile|tile||
|89|GatherND|gather_nd||
|90|NonMaxSuppression|non_max_suppression|WIP. Only available for batch size 1. To simplify post-processing ignore all OPs after non_max_suppression.|
|91|Gelu|gelu||
|92|NormalizeL2|tf.math.add, tf.math.l2_normalize|x/sqrt(max(sum(x\*\*2), eps)) or x/sqrt(add(sum(x\*\*2), eps))|
|93|Result|Identity|Output|

## 4. Setup
### 4-1. **[Environment construction pattern 1]** Execution by Docker (`strongly recommended`)
You do not need to install any packages other than Docker.
```bash
$ docker pull pinto0309/openvino2tensorflow
or
$ docker build -t pinto0309/openvino2tensorflow:latest .

# If you don't need to access the GUI of the HostPC and the USB camera.
$ docker run -it --rm \
  -v `pwd`:/home/user/workdir \
  pinto0309/openvino2tensorflow:latest

# If conversion to TF-TRT is not required. And if you need to access the HostPC GUI and USB camera.
$ xhost +local: && \
  docker run -it --rm \
  -v `pwd`:/home/user/workdir \
  -v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
  --device /dev/video0:/dev/video0:mwr \
  --net=host \
  -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  -e DISPLAY=$DISPLAY \
  --privileged \
  pinto0309/openvino2tensorflow:latest
$ cd workdir

# If you need to convert to TF-TRT. And if you need to access the HostPC GUI and USB camera.
$ xhost +local: && \
  docker run --gpus all -it --rm \
  -v `pwd`:/home/user/workdir \
  -v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
  --device /dev/video0:/dev/video0:mwr \
  --net=host \
  -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  -e DISPLAY=$DISPLAY \
  --privileged \
  pinto0309/openvino2tensorflow:latest
$ cd workdir

# If you are using iGPU (OpenCL). And if you need to access the HostPC GUI and USB camera.
$ xhost +local: && \
  docker run -it --rm \
  -v `pwd`:/home/user/workdir \
  -v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
  --device /dev/video0:/dev/video0:mwr \
  --net=host \
  -e LIBVA_DRIVER_NAME=iHD \
  -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  -e DISPLAY=$DISPLAY \
  --privileged \
  pinto0309/openvino2tensorflow:latest
$ cd workdir
```
### 4-2. **[Environment construction pattern 2]** Execution by Host machine
To install using the **[Python Package Index (PyPI)](https://pypi.org/project/openvino2tensorflow/)**, use the following command.

```bash
$ pip3 install --user --upgrade openvino2tensorflow
```

To install with the latest source code of the main branch, use the following command.

```bash
$ pip3 install --user --upgrade git+https://github.com/PINTO0309/openvino2tensorflow
```

## 5. Usage
### 5-1. openvino to tensorflow convert
```bash
usage: openvino2tensorflow
  [-h]
  --model_path MODEL_PATH
  [--model_output_path MODEL_OUTPUT_PATH]
  [--output_saved_model]
  [--output_h5]
  [--output_weight_and_json]
  [--output_pb]
  [--output_no_quant_float32_tflite]
  [--output_weight_quant_tflite]
  [--output_float16_quant_tflite]
  [--output_integer_quant_tflite]
  [--output_full_integer_quant_tflite]
  [--output_integer_quant_type OUTPUT_INTEGER_QUANT_TYPE]
  [--string_formulas_for_normalization STRING_FORMULAS_FOR_NORMALIZATION]
  [--calib_ds_type CALIB_DS_TYPE]
  [--ds_name_for_tfds_for_calibration DS_NAME_FOR_TFDS_FOR_CALIBRATION]
  [--split_name_for_tfds_for_calibration SPLIT_NAME_FOR_TFDS_FOR_CALIBRATION]
  [--download_dest_folder_path_for_the_calib_tfds DOWNLOAD_DEST_FOLDER_PATH_FOR_THE_CALIB_TFDS]
  [--tfds_download_flg]
  [--load_dest_file_path_for_the_calib_npy LOAD_DEST_FILE_PATH_FOR_THE_CALIB_NPY]
  [--output_tfjs]
  [--output_tftrt]
  [--output_coreml]
  [--output_edgetpu]
  [--output_onnx]
  [--onnx_opset ONNX_OPSET]
  [--output_myriad]
  [--vpu_number_of_shaves VPU_NUMBER_OF_SHAVES]
  [--vpu_number_of_cmx_slices VPU_NUMBER_OF_CMX_SLICES]
  [--replace_swish_and_hardswish]
  [--optimizing_hardswish_for_edgetpu]
  [--replace_prelu_and_minmax]
  [--yolact]
  [--restricted_resize_image_mode]
  [--weight_replacement_config WEIGHT_REPLACEMENT_CONFIG]
  [--use_experimental_new_quantizer]
  [--debug]
  [--debug_layer_number DEBUG_LAYER_NUMBER]


optional arguments:
  -h, --help
                        show this help message and exit
  --model_path MODEL_PATH
                        input IR model path (.xml)
  --model_output_path MODEL_OUTPUT_PATH
                        The output folder path of the converted model file
  --output_saved_model
                        saved_model output switch
  --output_h5
                        .h5 output switch
  --output_weight_and_json
                        weight of h5 and json output switch
  --output_pb
                        .pb output switch
  --output_no_quant_float32_tflite
                        float32 tflite output switch
  --output_weight_quant_tflite
                        weight quant tflite output switch
  --output_float16_quant_tflite
                        float16 quant tflite output switch
  --output_integer_quant_tflite
                        integer quant tflite output switch
  --output_full_integer_quant_tflite
                        full integer quant tflite output switch
  --output_integer_quant_type OUTPUT_INTEGER_QUANT_TYPE
                        Input and output types when doing Integer Quantization
                        ('int8 (default)' or 'uint8')
  --string_formulas_for_normalization STRING_FORMULAS_FOR_NORMALIZATION
                        String formulas for normalization. It is evaluated by
                        Pythons eval() function.
                        Default: '(data - [127.5,127.5,127.5]) / [127.5,127.5,127.5]'
  --calib_ds_type CALIB_DS_TYPE
                        Types of data sets for calibration. tfds or numpy
                        Default: numpy
  --ds_name_for_tfds_for_calibration DS_NAME_FOR_TFDS_FOR_CALIBRATION
                        Dataset name for TensorFlow Datasets for calibration.
                        https://www.tensorflow.org/datasets/catalog/overview
  --split_name_for_tfds_for_calibration SPLIT_NAME_FOR_TFDS_FOR_CALIBRATION
                        Split name for TensorFlow Datasets for calibration.
                        https://www.tensorflow.org/datasets/catalog/overview
  --download_dest_folder_path_for_the_calib_tfds DOWNLOAD_DEST_FOLDER_PATH_FOR_THE_CALIB_TFDS
                        Download destination folder path for the calibration
                        dataset. Default: $HOME/TFDS
  --tfds_download_flg
                        True to automatically download datasets from
                        TensorFlow Datasets. True or False
  --load_dest_file_path_for_the_calib_npy LOAD_DEST_FILE_PATH_FOR_THE_CALIB_NPY
                        The path from which to load the .npy file containing
                        the numpy binary version of the calibration data.
                        Default: sample_npy/calibration_data_img_sample.npy
  --output_tfjs
                        tfjs model output switch
  --output_tftrt
                        tftrt model output switch
  --output_coreml
                        coreml model output switch
  --output_edgetpu
                        edgetpu model output switch
  --output_onnx
                        onnx model output switch
  --onnx_opset ONNX_OPSET
                        onnx opset version number
  --output_myriad
                        myriad inference engine blob output switch
  --vpu_number_of_shaves VPU_NUMBER_OF_SHAVES
                        vpu number of shaves. Default: 4
  --vpu_number_of_cmx_slices VPU_NUMBER_OF_CMX_SLICES
                        vpu number of cmx slices. Default: 4
  --replace_swish_and_hardswish
                        Replace swish and hard-swish with each other
  --optimizing_hardswish_for_edgetpu
                        Optimizing hardswish for edgetpu
  --replace_prelu_and_minmax
                        Replace prelu and minimum/maximum with each other
  --yolact
                        Specify when converting the Yolact model
  --restricted_resize_image_mode
                        Specify this if the upsampling contains OPs that are
                        not scaled by integer multiples. Optimization for
                        EdgeTPU will be disabled.
  --weight_replacement_config WEIGHT_REPLACEMENT_CONFIG
                        Replaces the value of Const for each layer_id defined
                        in json. Specify the path to the json file.
                        'weight_replacement_config.json'
  --use_experimental_new_quantizer
                        Use MLIR\'s new quantization feature during INT8 quantization
                        in TensorFlowLite.
  --debug
                        debug mode switch
  --debug_layer_number DEBUG_LAYER_NUMBER
                        The last layer number to output when debugging. Used
                        only when --debug=True
```
### 5-2. saved_model to tflite convert
```bash
usage: saved_model_to_tflite
  [-h]
  --saved_model_dir_path SAVED_MODEL_DIR_PATH
  [--signature_def SIGNATURE_DEF]
  [--input_shapes INPUT_SHAPES]
  [--model_output_dir_path MODEL_OUTPUT_DIR_PATH]
  [--output_no_quant_float32_tflite]
  [--output_weight_quant_tflite]
  [--output_float16_quant_tflite]
  [--output_integer_quant_tflite]
  [--output_full_integer_quant_tflite]
  [--output_integer_quant_type OUTPUT_INTEGER_QUANT_TYPE]
  [--string_formulas_for_normalization STRING_FORMULAS_FOR_NORMALIZATION]
  [--calib_ds_type CALIB_DS_TYPE]
  [--ds_name_for_tfds_for_calibration DS_NAME_FOR_TFDS_FOR_CALIBRATION]
  [--split_name_for_tfds_for_calibration SPLIT_NAME_FOR_TFDS_FOR_CALIBRATION]
  [--download_dest_folder_path_for_the_calib_tfds DOWNLOAD_DEST_FOLDER_PATH_FOR_THE_CALIB_TFDS]
  [--tfds_download_flg]
  [--load_dest_file_path_for_the_calib_npy LOAD_DEST_FILE_PATH_FOR_THE_CALIB_NPY]
  [--output_tfjs]
  [--output_tftrt]
  [--output_coreml]
  [--output_edgetpu]
  [--output_onnx]
  [--onnx_opset ONNX_OPSET]
  [--use_experimental_new_quantizer]

optional arguments:
  -h, --help
                        show this help message and exit
  --saved_model_dir_path SAVED_MODEL_DIR_PATH
                        Input saved_model dir path
  --signature_def SIGNATURE_DEF
                        Specifies the signature name to load from saved_model
  --input_shapes INPUT_SHAPES
                        Overwrites an undefined input dimension (None or -1).
                        Specify the input shape in [n,h,w,c] format.
                        For non-4D tensors, specify [a,b,c,d,e], [a,b], etc.
                        A comma-separated list if there are multiple inputs.
                        (e.g.) --input_shapes [1,256,256,3],[1,64,64,3],[1,2,16,16,3]
  --model_output_dir_path MODEL_OUTPUT_DIR_PATH
                        The output folder path of the converted model file
  --output_no_quant_float32_tflite
                        float32 tflite output switch
  --output_weight_quant_tflite
                        weight quant tflite output switch
  --output_float16_quant_tflite
                        float16 quant tflite output switch
  --output_integer_quant_tflite
                        integer quant tflite output switch
  --output_full_integer_quant_tflite
                        full integer quant tflite output switch
  --output_integer_quant_type OUTPUT_INTEGER_QUANT_TYPE
                        Input and output types when doing Integer Quantization
                        ('int8 (default)' or 'uint8')
  --string_formulas_for_normalization STRING_FORMULAS_FOR_NORMALIZATION
                        String formulas for normalization. It is evaluated by
                        Pythons eval() function.
                        Default: '(data - [127.5,127.5,127.5]) / [127.5,127.5,127.5]'
  --calib_ds_type CALIB_DS_TYPE
                        Types of data sets for calibration. tfds or numpy
                        Default: numpy
  --ds_name_for_tfds_for_calibration DS_NAME_FOR_TFDS_FOR_CALIBRATION
                        Dataset name for TensorFlow Datasets for calibration.
                        https://www.tensorflow.org/datasets/catalog/overview
  --split_name_for_tfds_for_calibration SPLIT_NAME_FOR_TFDS_FOR_CALIBRATION
                        Split name for TensorFlow Datasets for calibration.
                        https://www.tensorflow.org/datasets/catalog/overview
  --download_dest_folder_path_for_the_calib_tfds DOWNLOAD_DEST_FOLDER_PATH_FOR_THE_CALIB_TFDS
                        Download destination folder path for the calibration
                        dataset. Default: $HOME/TFDS
  --tfds_download_flg
                        True to automatically download datasets from
                        TensorFlow Datasets. True or False
  --load_dest_file_path_for_the_calib_npy LOAD_DEST_FILE_PATH_FOR_THE_CALIB_NPY
                        The path from which to load the .npy file containing
                        the numpy binary version of the calibration data.
                        Default: sample_npy/calibration_data_img_sample.npy
  --output_tfjs
                        tfjs model output switch
  --output_tftrt
                        tftrt model output switch
  --output_coreml
                        coreml model output switch
  --output_edgetpu
                        edgetpu model output switch
  --output_onnx
                        onnx model output switch
  --onnx_opset ONNX_OPSET
                        onnx opset version number
  --use_experimental_new_quantizer
                        Use MLIR\'s new quantization feature during INT8 quantization
                        in TensorFlowLite.
```
### 5-3. pb to saved_model convert
```bash
usage: pb_to_saved_model
  [-h]
  --pb_file_path PB_FILE_PATH
  --inputs INPUTS
  --outputs OUTPUTS
  [--model_output_path MODEL_OUTPUT_PATH]

optional arguments:
  -h, --help
                        show this help message and exit
  --pb_file_path PB_FILE_PATH
                        Input .pb file path (.pb)
  --inputs INPUTS
                        (e.g.1) input:0,input:1,input:2
                        (e.g.2) images:0,input:0,param:0
  --outputs OUTPUTS
                        (e.g.1) output:0,output:1,output:2
                        (e.g.2) Identity:0,Identity:1,output:0
  --model_output_path MODEL_OUTPUT_PATH
                        The output folder path of the converted model file
```
### 5-4. pb to tflite convert
```bash
usage: pb_to_tflite
  [-h]
  --pb_file_path PB_FILE_PATH
  --inputs INPUTS
  --outputs OUTPUTS
  [--model_output_path MODEL_OUTPUT_PATH]

optional arguments:
  -h, --help
                        show this help message and exit
  --pb_file_path PB_FILE_PATH
                        Input .pb file path (.pb)
  --inputs INPUTS
                        (e.g.1) input,input_1,input_2
                        (e.g.2) images,input,param
  --outputs OUTPUTS
                        (e.g.1) output,output_1,output_2
                        (e.g.2) Identity,Identity_1,output
  --model_output_path MODEL_OUTPUT_PATH
                        The output folder path of the converted model file
```
### 5-5. saved_model to pb convert
```bash
usage: saved_model_to_pb
  [-h]
  --saved_model_dir_path SAVED_MODEL_DIR_PATH
  [--model_output_dir_path MODEL_OUTPUT_DIR_PATH]
  [--signature_name SIGNATURE_NAME]

optional arguments:
  -h, --help
                        show this help message and exit
  --saved_model_dir_path SAVED_MODEL_DIR_PATH
                        Input saved_model dir path
  --model_output_dir_path MODEL_OUTPUT_DIR_PATH
                        The output folder path of the converted model file (.pb)
  --signature_name SIGNATURE_NAME
                        Signature name to be extracted from saved_model
```
### 5-6. Extraction of IR weight
```bash
usage: ir_weight_extractor
  [-h]
  -m MODEL
  -o OUTPUT_PATH

optional arguments:
  -h, --help
                        show this help message and exit
  -m MODEL, --model MODEL
                        input IR model path
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        weights output folder path
```

## 6. Execution sample
### 6-1. Conversion of OpenVINO IR to Tensorflow models
OutOfMemory may occur when converting to saved_model or h5 when the file size of the original model is large, please try the conversion to a pb file alone.
```
$ openvino2tensorflow \
  --model_path openvino/448x448/FP32/Resnet34_3inputs_448x448_20200609.xml \
  --output_saved_model \
  --output_pb \
  --output_weight_quant_tflite \
  --output_float16_quant_tflite \
  --output_no_quant_float32_tflite
```
### 6-2. Convert Protocol Buffer (.pb) to saved_model
This tool is useful if you want to check the internal structure of pb files, tflite files, .h5 files, coreml files and IR (.xml) files. **https://lutzroeder.github.io/netron/**
```
$ pb_to_saved_model \
  --pb_file_path model_float32.pb \
  --inputs inputs:0 \
  --outputs Identity:0
```
### 6-3. Convert Protocol Buffer (.pb) to tflite
```
$ pb_to_tflite \
  --pb_file_path model_float32.pb \
  --inputs inputs \
  --outputs Identity,Identity_1,Identity_2
```
### 6-4. Convert saved_model to Protocol Buffer (.pb)
```
$ saved_model_to_pb \
  --saved_model_dir_path saved_model \
  --model_output_dir_path pb_from_saved_model \
  --signature_name serving_default
```

### 6-5. Converts saved_model to OpenVINO IR
```
$ python3 ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo_tf.py \
  --saved_model_dir saved_model \
  --output_dir openvino/reverse
```
### 6-6. Checking the structure of saved_model
```
$ saved_model_cli show \
  --dir saved_model \
  --tag_set serve \
  --signature_def serving_default
```

### 6-7. Replace weights or constant values in **`Const`** OP
#### 6-7-1. Overview
If the transformation behavior of **`Reshape`**, **`Transpose`**, etc. does not go as expected, you can force the **`Const`** content to change by defining weights and constant values in a JSON file and having it read in.
```
$ openvino2tensorflow \
  --model_path xxx.xml \
  --output_saved_model \
  --output_pb \
  --output_weight_quant_tflite \
  --output_float16_quant_tflite \
  --output_no_quant_float32_tflite \
  --weight_replacement_config weight_replacement_config_sample.json
```
Structure of JSON sample
```json
{
    "format_version": 1,
    "layers": [
        {
            "layer_id": "1123",
            "replace_mode": "direct",
            "values": [
                1,
                2,
                513,
                513
            ]
        },
        {
            "layer_id": "1125",
            "replace_mode": "npy",
            "values": "weights_sample/1125.npy"
        }
    ]
}
```

|No.|Elements|Description|
|:--|:--|:--|
|1|format_version|Format version of weight_replacement_config. Only 1 so far.|
|2|layers|A list of layers. Enclose it with "[ ]" to define multiple layers to child elements.|
|2-1|layer_id|ID of the Const layer whose weight/constant parameter is to be swapped. For example, specify "1123" for layer id="1123" for type="Const" in .xml.<br>![Screenshot 2021-02-08 01:06:30](https://user-images.githubusercontent.com/33194443/107152221-068a0f00-69aa-11eb-9d9e-f48bb1c3f781.png)|
|2-2|replace_mode|"direct" or "npy".<br>"direct": Specify the values of the Numpy matrix directly in the "values" attribute. Ignores the values recorded in the .bin file and replaces them with the values specified in "values".<br>![Screenshot 2021-02-08 01:12:06](https://user-images.githubusercontent.com/33194443/107152361-cc6d3d00-69aa-11eb-8302-5e18a723ec34.png)<br>"npy": Load a Numpy binary file with the matrix output by **`np.save('xyz', a)`**. The "values" attribute specifies the path to the Numpy binary file.<br>![Screenshot 2021-02-08 01:12:23](https://user-images.githubusercontent.com/33194443/107152376-dc851c80-69aa-11eb-9b3f-469b91af1d19.png)|
|2-3|values|Specify the value or the path to the Numpy binary file to replace the weight/constant value recorded in .bin. The way to specify is as described in the description of 'replace_mode'.|

#### 6-7-2. Example
  - YOLOX Nano 320x320 (NCHW format)
  - yolox_nano_320x320.xml
  - yolox_nano_320x320.bin
  1. Let's assume that you don't need **`Transpose`** in the final layer of the model. Here you have **`[1, 85, 2100]`** as input, and the original OpenVINO model transposes **`[0, 2, 1]`** in that order to obtain the tensor **`[1, 2100, 85]`**. This figure shows the visualization of a **`yolox_nano_320x320.xml`** file using **[Netron](https://netron.app/)**. The number shown in the **`OUTPUTS`** - **`output`** - **`name:`** is the layer ID of **`Transpose`**. The layer ID 660 is the number in the part before the colon. The number in the part after the colon is called the port number 2. However, what you are trying to change is the transposition parameter of the **`INPUTS`** - **`custom`** - **`name:`** part. The name of the parameter you are trying to change is **`625`**. Note that **`625`** is not a layer ID, just a name.
![Screenshot 2021-08-04 23:45:15](https://user-images.githubusercontent.com/33194443/128202697-1e9a5110-3482-424c-8c03-202023478571.png)
  2. Check the model structure as recorded in .xml. First, open **`yolox_nano_320x320.xml`** in your favorite IDE.
![Screenshot 2021-08-05 00:00:38](https://user-images.githubusercontent.com/33194443/128204779-3a446618-c3f4-4968-ab3d-50648652da53.png)
![Screenshot 2021-08-05 00:08:50](https://user-images.githubusercontent.com/33194443/128205851-c4effc4a-8033-49a4-887f-4af0829824b9.png)
  3. Search for **`to-layer="660"`** (Transpose) in the IDE. In the figure below, Layer ID **`658`** and Layer ID **`659`** are represented as input values connected to Layer ID **`660`**.
![Screenshot 2021-08-05 00:17:31](https://user-images.githubusercontent.com/33194443/128207323-400f5145-46fd-4734-b186-408d4a8cc7d0.png)
In the figure below, one of them is **`658`** and one of them is **`659`**. It is difficult to determine exactly what it is from the image alone. You must again note that **`658:3`** in the image is only a name, not a layer ID. It is worth noting here that the type of value you want to replace is **`Const`**.
![Screenshot 2021-08-05 00:26:29](https://user-images.githubusercontent.com/33194443/128208774-1dd27e57-e453-4942-8708-c118d5cec10c.png)
  4. Now you will search for layer ID **`"658"`** in the IDE. The type is **`"Concat"`**, so the desired layer was not this one. What you are looking for is **`"Const"`**.
![Screenshot 2021-08-05 01:02:00](https://user-images.githubusercontent.com/33194443/128214658-ec28bbc5-685b-4f92-b0ca-ca6e5389194c.png)
  5. Now, search for layer ID **`659`** in the IDE. The type is **`"Const"`**. Now you can finally identify that the layer ID of the layer you want to replace is **`659`**.
![Screenshot 2021-08-05 01:05:33](https://user-images.githubusercontent.com/33194443/128215161-9631100f-0bff-4c49-ad57-9f3a3e8ad7be.png)
  6. Create a JSON file to replace the constants **`[0, 2, 1]`** with **`[0, 1, 2]`**, and you can use any name for the JSON file. Suppose you save the file with the name **`replace.json`**. If you want to replace it with a numpy matrix, specify **`"npy"`** for **`"replace_mode":`** and the path to the **`.npy`** file for **`"values":`**.
  ```json
  {
    "format_version": 1,
    "layers": [
        {
            "layer_id": "659",
            "replace_mode": "direct",
            "values": [
                0,
                1,
                2
            ]
        }
    ]
  }
  ```
  ```json
  {
    "format_version": 1,
    "layers": [
        {
            "layer_id": "659",
            "replace_mode": "npy",
            "values": "path/to/your/xxx.npy"
        }
    ]
  }
  ```
  7. Specify the created JSON file as the argument of the **`--weight_replacement_config`** parameter of the conversion command and execute it. This is the end of the explanation of how to replace weights and constants.
  ```
  $ openvino2tensorflow \
  --model_path yolox_nano_320x320.xml \
  --output_saved_model \
  --output_pb \
  --output_no_quant_float32_tflite \
  --weight_replacement_config replace.json
  ```


### 6-8. Check the contents of the .npy file, which is a binary version of the image file
```
$ view_npy --npy_file_path sample_npy/calibration_data_img_sample.npy
```
Press the **`Q`** button to display the next image. **`calibration_data_img_sample.npy`** contains 20 images extracted from the MS-COCO data set.
![ezgif com-gif-maker](https://user-images.githubusercontent.com/33194443/109318923-aba15480-7891-11eb-84aa-034f77125f34.gif)

### 6-9. Sample image of a conversion error message
Since it is very difficult to mechanically predict the correct behavior of **`Transpose`** and **`Reshape`**, errors like the one shown below may occur. Using the information in the figure below, try several times to force the replacement of constants and weights using the **`--weight_replacement_config`** option [#6-7-replace-weights-or-constant-values-in-const-op](#6-7-replace-weights-or-constant-values-in-const-op). This is a very patient process, but if you take the time, you should be able to convert it correctly.
![error_sample2](https://user-images.githubusercontent.com/33194443/124498169-e181b700-ddf6-11eb-9200-83ba44c62410.png)

## 7. Output sample
![Screenshot 2020-10-16 00:08:40](https://user-images.githubusercontent.com/33194443/96149093-e38fa700-0f43-11eb-8101-65fc20b2cc8f.png)


## 8. Model Structure
**[https://digital-standard.com/threedpose/models/Resnet34_3inputs_448x448_20200609.onnx](https://github.com/digital-standard/ThreeDPoseUnityBarracuda#download-and-put-files)**

|ONNX (NCHW)|OpenVINO (NCHW)|TFLite (NHWC)|
|:--:|:--:|:--:|
|![Resnet34_3inputs_448x448_20200609 onnx_](https://user-images.githubusercontent.com/33194443/96398683-62683680-1207-11eb-928d-e4cb6c8cc188.png)|![Resnet34_3inputs_448x448_20200609 xml](https://user-images.githubusercontent.com/33194443/96153010-23f12400-0f48-11eb-8186-4bbad73b517a.png)|![model_float32 tflite](https://user-images.githubusercontent.com/33194443/96153019-26ec1480-0f48-11eb-96be-0c405ee2cbf7.png)|

## 9. My article
- **[[English] Converting PyTorch, ONNX, Caffe, and OpenVINO (NCHW) models to Tensorflow / TensorflowLite (NHWC) in a snap](https://qiita.com/PINTO/items/ed06e03eb5c007c2e102)**

- **[PyTorch, ONNX, Caffe, OpenVINO (NCHW) のモデルをTensorflow / TensorflowLite (NHWC) へお手軽に変換する](https://qiita.com/PINTO/items/7a0bcaacc77bb5d6abb1)**

- **[tf.image.resizeを含むFull Integer Quantization (.tflite)モデルのEdgeTPUモデルへの変換後の推論時に発生する "main.ERROR - Only float32 and uint8 are supported currently, got -xxx.Node number n (op name) failed to invoke" エラーの回避方法](https://qiita.com/PINTO/items/6ff62da1d02089442c8c)**

## 10. Conversion Confirmed Models
1. u-2-net
2. mobilenet-v2-pytorch
3. midasnet
4. footprints
5. efficientnet-b0-pytorch
6. efficientdet-d0
7. dense_depth
8. deeplabv3
9. colorization-v2-norebal
10. age-gender-recognition-retail-0013
11. resnet
12. arcface
13. emotion-ferplus
14. mosaic
15. retinanet
16. shufflenet-v2
17. squeezenet
18. version-RFB-320
19. yolov4
20. yolov4x-mish
21. ThreeDPoseUnityBarracuda - Resnet34_3inputs_448x448
22. efficientnet-lite4
23. nanodet
24. yolov4-tiny
25. yolov5s
26. yolact
27. MiDaS v2
28. MODNet
29. Person Reidentification
30. DeepSort
31. DINO (Transformer)
