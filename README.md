# openvino2tensorflow

<p align="center">
  <img src="https://user-images.githubusercontent.com/33194443/104584047-4e688f80-56a5-11eb-8dc2-5816487239d0.png" />
</p>

This script converts the OpenVINO IR model to Tensorflow's saved_model, tflite, h5, TensorFlow.js, TF-TRT(TensorRT), CoreML, EdgeTPU and pb. And the conversion from .pb to saved_model and from saved_model to .pb and from .pb to .tflite and from saved_model to tflite.

Work in progress now.

**I'm continuing to add more layers of support and bug fixes on a daily basis. If you have a model that you are having trouble converting, please share the `.bin` and `.xml` with the issue. I will try to convert as much as possible.**
  
[![PyPI - Downloads](https://img.shields.io/pypi/dm/openvino2tensorflow?color=2BAF2B&label=Downloads%EF%BC%8FInstalled)](https://pypistats.org/packages/openvino2tensorflow) ![GitHub](https://img.shields.io/github/license/PINTO0309/openvino2tensorflow?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/openvino2tensorflow?color=2BAF2B)](https://pypi.org/project/openvino2tensorflow/)
  
![ezgif com-gif-maker (4)](https://user-images.githubusercontent.com/33194443/103457894-4ffd9380-4d46-11eb-86dd-f753f2fca093.gif)
  
![ezgif com-gif-maker (3)](https://user-images.githubusercontent.com/33194443/103456348-8d5b2480-4d38-11eb-8a58-9b7c7203b18c.gif)
  
## 1. Environment
- TensorFlow v2.3.1+
- OpenVINO 2021.1.110+
- Python 3.6+

## 2. Use case

- PyTorch (NCHW) -> ONNX (NCHW) -> OpenVINO (NCHW) ->
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TFLite (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TFJS (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TF-TRT (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> EdgeTPU (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> CoreML (NHWC)

- Caffe (NCHW) -> OpenVINO (NCHW) ->
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TFLite (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TFJS (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TF-TRT (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> EdgeTPU (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> CoreML (NHWC)

- MXNet (NCHW) -> OpenVINO (NCHW) ->
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TFLite (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TFJS (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TF-TRT (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> EdgeTPU (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> CoreML (NHWC)

- Keras (NHWC) -> OpenVINO (NCHW・Optimized) ->
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TFLite (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TFJS (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TF-TRT (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> EdgeTPU (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> CoreML (NHWC)

- saved_model -> **`saved_model_to_pb`** -> pb

- saved_model -> **`saved_model_to_tflite`** -> tflite

- pb -> **`pb_to_tflite`** -> tflite

- pb -> **`pb_to_saved_model`** -> saved_model

## 3. Supported Layers
- Currently, only 4D tensors are supported as input tensors.
- Currently, there are problems with the Reshape operation of 5D Tensor.

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
|87|FakeQuantize|subtract, multiply, round, greater, where, less_equal, add|experimental|
|88|Result|Identity|Output|

## 4. Setup

To install using the **[Python Package Index (PyPI)](https://pypi.org/project/openvino2tensorflow/)**, use the following command.

```bash
$ pip3 install openvino2tensorflow --upgrade
```

To install with the latest source code of the main branch, use the following command.

```bash
$ pip3 install git+https://github.com/PINTO0309/openvino2tensorflow --upgrade
```

## 5. Usage
### 5-1. openvino to tensorflow convert
```bash
usage: openvino2tensorflow [-h] --model_path MODEL_PATH
                           [--model_output_path MODEL_OUTPUT_PATH]
                           [--output_saved_model OUTPUT_SAVED_MODEL]
                           [--output_h5 OUTPUT_H5]
                           [--output_weight_and_json OUTPUT_WEIGHT_AND_JSON]
                           [--output_pb OUTPUT_PB]
                           [--output_no_quant_float32_tflite OUTPUT_NO_QUANT_FLOAT32_TFLITE]
                           [--output_weight_quant_tflite OUTPUT_WEIGHT_QUANT_TFLITE]
                           [--output_float16_quant_tflite OUTPUT_FLOAT16_QUANT_TFLITE]
                           [--output_integer_quant_tflite OUTPUT_INTEGER_QUANT_TFLITE]
                           [--output_full_integer_quant_tflite OUTPUT_FULL_INTEGER_QUANT_TFLITE]
                           [--output_integer_quant_type OUTPUT_INTEGER_QUANT_TYPE]
                           [--string_formulas_for_normalization STRING_FORMULAS_FOR_NORMALIZATION]
                           [--calib_ds_type CALIB_DS_TYPE]
                           [--ds_name_for_tfds_for_calibration DS_NAME_FOR_TFDS_FOR_CALIBRATION]
                           [--split_name_for_tfds_for_calibration SPLIT_NAME_FOR_TFDS_FOR_CALIBRATION]
                           [--download_dest_folder_path_for_the_calib_tfds DOWNLOAD_DEST_FOLDER_PATH_FOR_THE_CALIB_TFDS]
                           [--tfds_download_flg TFDS_DOWNLOAD_FLG]
                           [--output_tfjs OUTPUT_TFJS]
                           [--output_tftrt OUTPUT_TFTRT]
                           [--output_coreml OUTPUT_COREML]
                           [--output_edgetpu OUTPUT_EDGETPU]
                           [--replace_swish_and_hardswish REPLACE_SWISH_AND_HARDSWISH]
                           [--optimizing_hardswish_for_edgetpu OPTIMIZING_HARDSWISH_FOR_EDGETPU]
                           [--replace_prelu_and_minmax REPLACE_PRELU_AND_MINMAX]
                           [--yolact] [--debug]
                           [--debug_layer_number DEBUG_LAYER_NUMBER]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        input IR model path (.xml)
  --model_output_path MODEL_OUTPUT_PATH
                        The output folder path of the converted model file
  --output_saved_model OUTPUT_SAVED_MODEL
                        saved_model output switch
  --output_h5 OUTPUT_H5
                        .h5 output switch
  --output_weight_and_json OUTPUT_WEIGHT_AND_JSON
                        weight of h5 and json output switch
  --output_pb OUTPUT_PB
                        .pb output switch
  --output_no_quant_float32_tflite OUTPUT_NO_QUANT_FLOAT32_TFLITE
                        float32 tflite output switch
  --output_weight_quant_tflite OUTPUT_WEIGHT_QUANT_TFLITE
                        weight quant tflite output switch
  --output_float16_quant_tflite OUTPUT_FLOAT16_QUANT_TFLITE
                        float16 quant tflite output switch
  --output_integer_quant_tflite OUTPUT_INTEGER_QUANT_TFLITE
                        integer quant tflite output switch
  --output_full_integer_quant_tflite OUTPUT_FULL_INTEGER_QUANT_TFLITE
                        full integer quant tflite output switch
  --output_integer_quant_type OUTPUT_INTEGER_QUANT_TYPE
                        Input and output types when doing Integer Quantization
                        ('int8 (default)' or 'uint8')
  --string_formulas_for_normalization STRING_FORMULAS_FOR_NORMALIZATION
                        String formulas for normalization. It is evaluated by
                        Pythons eval() function.
                        Default: '(data - [127.5,127.5,127.5]) / [127.5,127.5,127.5]'
  --calib_ds_type CALIB_DS_TYPE
                        Types of data sets for calibration. tfds or
                        numpy(Future Implementation)
  --ds_name_for_tfds_for_calibration DS_NAME_FOR_TFDS_FOR_CALIBRATION
                        Dataset name for TensorFlow Datasets for calibration.
                        https://www.tensorflow.org/datasets/catalog/overview
  --split_name_for_tfds_for_calibration SPLIT_NAME_FOR_TFDS_FOR_CALIBRATION
                        Split name for TensorFlow Datasets for calibration.
                        https://www.tensorflow.org/datasets/catalog/overview
  --download_dest_folder_path_for_the_calib_tfds DOWNLOAD_DEST_FOLDER_PATH_FOR_THE_CALIB_TFDS
                        Download destination folder path for the calibration
                        dataset. Default: $HOME/TFDS
  --tfds_download_flg TFDS_DOWNLOAD_FLG
                        True to automatically download datasets from
                        TensorFlow Datasets. True or False
  --output_tfjs OUTPUT_TFJS
                        tfjs model output switch
  --output_tftrt OUTPUT_TFTRT
                        tftrt model output switch
  --output_coreml OUTPUT_COREML
                        coreml model output switch
  --output_edgetpu OUTPUT_EDGETPU
                        edgetpu model output switch
  --replace_swish_and_hardswish REPLACE_SWISH_AND_HARDSWISH
                        Replace swish and hard-swish with each other
  --optimizing_hardswish_for_edgetpu OPTIMIZING_HARDSWISH_FOR_EDGETPU
                        Optimizing hardswish for edgetpu
  --replace_prelu_and_minmax REPLACE_PRELU_AND_MINMAX
                        Replace prelu and minimum/maximum with each other
  --yolact              Specify when converting the Yolact model
  --debug               debug mode switch
  --debug_layer_number DEBUG_LAYER_NUMBER
                        The last layer number to output when debugging. Used
                        only when --debug=True
```
### 5-2. saved_model to tflite convert
```bash
usage: saved_model_to_tflite [-h] --saved_model_dir_path
                             SAVED_MODEL_DIR_PATH
                             [--signature_def SIGNATURE_DEF]
                             [--input_shapes INPUT_SHAPES]
                             [--model_output_dir_path MODEL_OUTPUT_DIR_PATH]
                             [--output_no_quant_float32_tflite OUTPUT_NO_QUANT_FLOAT32_TFLITE]
                             [--output_weight_quant_tflite OUTPUT_WEIGHT_QUANT_TFLITE]
                             [--output_float16_quant_tflite OUTPUT_FLOAT16_QUANT_TFLITE]
                             [--output_integer_quant_tflite OUTPUT_INTEGER_QUANT_TFLITE]
                             [--output_full_integer_quant_tflite OUTPUT_FULL_INTEGER_QUANT_TFLITE]
                             [--output_integer_quant_type OUTPUT_INTEGER_QUANT_TYPE]
                             [--string_formulas_for_normalization STRING_FORMULAS_FOR_NORMALIZATION]
                             [--calib_ds_type CALIB_DS_TYPE]
                             [--ds_name_for_tfds_for_calibration DS_NAME_FOR_TFDS_FOR_CALIBRATION]
                             [--split_name_for_tfds_for_calibration SPLIT_NAME_FOR_TFDS_FOR_CALIBRATION]
                             [--download_dest_folder_path_for_the_calib_tfds DOWNLOAD_DEST_FOLDER_PATH_FOR_THE_CALIB_TFDS]
                             [--tfds_download_flg TFDS_DOWNLOAD_FLG]
                             [--output_tfjs OUTPUT_TFJS]
                             [--output_tftrt OUTPUT_TFTRT]
                             [--output_coreml OUTPUT_COREML]
                             [--output_edgetpu OUTPUT_EDGETPU]

optional arguments:
  -h, --help            show this help message and exit
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
  --output_no_quant_float32_tflite OUTPUT_NO_QUANT_FLOAT32_TFLITE
                        float32 tflite output switch
  --output_weight_quant_tflite OUTPUT_WEIGHT_QUANT_TFLITE
                        weight quant tflite output switch
  --output_float16_quant_tflite OUTPUT_FLOAT16_QUANT_TFLITE
                        float16 quant tflite output switch
  --output_integer_quant_tflite OUTPUT_INTEGER_QUANT_TFLITE
                        integer quant tflite output switch
  --output_full_integer_quant_tflite OUTPUT_FULL_INTEGER_QUANT_TFLITE
                        full integer quant tflite output switch
  --output_integer_quant_type OUTPUT_INTEGER_QUANT_TYPE
                        Input and output types when doing Integer Quantization
                        ('int8 (default)' or 'uint8')
  --string_formulas_for_normalization STRING_FORMULAS_FOR_NORMALIZATION
                        String formulas for normalization. It is evaluated by
                        Pythons eval() function.
                        Default: '(data - [127.5,127.5,127.5]) / [127.5,127.5,127.5]'
  --calib_ds_type CALIB_DS_TYPE
                        Types of data sets for calibration. tfds or
                        numpy(Future Implementation)
  --ds_name_for_tfds_for_calibration DS_NAME_FOR_TFDS_FOR_CALIBRATION
                        Dataset name for TensorFlow Datasets for calibration.
                        https://www.tensorflow.org/datasets/catalog/overview
  --split_name_for_tfds_for_calibration SPLIT_NAME_FOR_TFDS_FOR_CALIBRATION
                        Split name for TensorFlow Datasets for calibration.
                        https://www.tensorflow.org/datasets/catalog/overview
  --download_dest_folder_path_for_the_calib_tfds DOWNLOAD_DEST_FOLDER_PATH_FOR_THE_CALIB_TFDS
                        Download destination folder path for the calibration
                        dataset. Default: $HOME/TFDS
  --tfds_download_flg TFDS_DOWNLOAD_FLG
                        True to automatically download datasets from
                        TensorFlow Datasets. True or False
  --output_tfjs OUTPUT_TFJS
                        tfjs model output switch
  --output_tftrt OUTPUT_TFTRT
                        tftrt model output switch
  --output_coreml OUTPUT_COREML
                        coreml model output switch
  --output_edgetpu OUTPUT_EDGETPU
                        edgetpu model output switch
```
### 5-3. pb to saved_model convert
```bash
usage: pb_to_saved_model [-h] --pb_file_path PB_FILE_PATH
                         --inputs INPUTS
                         --outputs OUTPUTS
                         [--model_output_path MODEL_OUTPUT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --pb_file_path PB_FILE_PATH
                        Input .pb file path (.pb)
  --inputs INPUTS       (e.g.1) input:0,input:1,input:2
                        (e.g.2) images:0,input:0,param:0
  --outputs OUTPUTS     (e.g.1) output:0,output:1,output:2
                        (e.g.2) Identity:0,Identity:1,output:0
  --model_output_path MODEL_OUTPUT_PATH
                        The output folder path of the converted model file
```
### 5-4. pb to tflite convert
```bash
usage: pb_to_tflite [-h] --pb_file_path PB_FILE_PATH --inputs INPUTS
                    --outputs OUTPUTS
                    [--model_output_path MODEL_OUTPUT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --pb_file_path PB_FILE_PATH
                        Input .pb file path (.pb)
  --inputs INPUTS       (e.g.1) input,input_1,input_2
                        (e.g.2) images,input,param
  --outputs OUTPUTS     (e.g.1) output,output_1,output_2
                        (e.g.2) Identity,Identity_1,output
  --model_output_path MODEL_OUTPUT_PATH
                        The output folder path of the converted model file
```
### 5-5. saved_model to pb convert
```bash
usage: saved_model_to_pb [-h] --saved_model_dir_path SAVED_MODEL_DIR_PATH
                         [--model_output_dir_path MODEL_OUTPUT_DIR_PATH]
                         [--signature_name SIGNATURE_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --saved_model_dir_path SAVED_MODEL_DIR_PATH
                        Input saved_model dir path
  --model_output_dir_path MODEL_OUTPUT_DIR_PATH
                        The output folder path of the converted model file (.pb)
  --signature_name SIGNATURE_NAME
                        Signature name to be extracted from saved_model
```
### 5-6. Extraction of IR weight
```bash
usage: ir_weight_extractor [-h] -m MODEL -o OUTPUT_PATH

optional arguments:
  -h, --help            show this help message and exit
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
  --model_path=openvino/448x448/FP32/Resnet34_3inputs_448x448_20200609.xml \
  --output_saved_model True \
  --output_pb True \
  --output_weight_quant_tflite True \
  --output_float16_quant_tflite True \
  --output_no_quant_float32_tflite True
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

## 7. Output sample
![Screenshot 2020-10-16 00:08:40](https://user-images.githubusercontent.com/33194443/96149093-e38fa700-0f43-11eb-8101-65fc20b2cc8f.png)


## 8. Model Structure
**[https://digital-standard.com/threedpose/models/Resnet34_3inputs_448x448_20200609.onnx](https://github.com/digital-standard/ThreeDPoseUnityBarracuda#download-and-put-files)**  

|ONNX|OpenVINO|TFLite|
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
