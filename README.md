# openvino2tensorflow
This script converts the OpenVINO IR model to Tensorflow's saved_model, tflite, h5 and pb.

Work in progress now.

**I'm continuing to add more layers of support and bug fixes on a daily basis. If you have a model that you are having trouble converting, please share the `.bin` and `.xml` with the issue. I will try to convert as much as possible.**
  
[![PyPI - Downloads](https://img.shields.io/pypi/dm/openvino2tensorflow?color=2BAF2B&label=Downloads%EF%BC%8FInstalled)](https://pypistats.org/packages/openvino2tensorflow) ![GitHub](https://img.shields.io/github/license/PINTO0309/openvino2tensorflow?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/openvino2tensorflow?color=2BAF2B)](https://pypi.org/project/openvino2tensorflow/)
  
## 1. Environment
- TensorFlow v2.3.1
- OpenVINO 2021.1.110
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

## 3. Supported Layers
- Currently, only 4D tensors are supported as input tensors.
- Currently, there are problems with the Reshape operation of 5D Tensor.

|No.|OpenVINO Layer|TF Layer|Remarks|
|:--:|:--|:--|:--|
|1|Parameter|Input|Input (4D tensor only)|
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
|79|VariadicSplit|Split||
|80|MVN|reduce_mean, sqrt, reduce_variance|(x-reduce_mean(x))/sqrt(reduce_variance(x)+eps)|
|81|Result|Identity|Output|

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
                           [--replace_swish_and_hardswish REPLACE_SWISH_AND_HARDSWISH]
                           [--replace_prelu_and_minmax REPLACE_PRELU_AND_MINMAX]
                           [--debug]
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
  --replace_swish_and_hardswish REPLACE_SWISH_AND_HARDSWISH
                        Replace swish and hard-swish with each other.
  --replace_prelu_and_minmax REPLACE_PRELU_AND_MINMAX
                        Replace prelu and minimum/maximum with each other.
  --debug               debug mode switch
  --debug_layer_number DEBUG_LAYER_NUMBER
                        The last layer number to output when debugging. Used
                        only when --debug=True.
```
```bash
usage: pb_to_saved_model [-h] --pb_file_path PB_FILE_PATH
                         --inputs INPUTS
                         --outputs OUTPUTS
                         [--model_output_path MODEL_OUTPUT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --pb_file_path PB_FILE_PATH
                        Input .pb file path (.pb)
  --inputs INPUTS       (e.g.1) input:0,input:1,input:2 / (e.g.2)
                        images:0,input:0,param:0
  --outputs OUTPUTS     (e.g.1) output:0,output:1,output:2 / (e.g.2)
                        Identity:0,Identity:1,output:0
  --model_output_path MODEL_OUTPUT_PATH
                        The output folder path of the converted model file
```
```bash
usage: pb_to_tflite.py [-h] --pb_file_path PB_FILE_PATH --inputs INPUTS
                       --outputs OUTPUTS
                       [--model_output_path MODEL_OUTPUT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --pb_file_path PB_FILE_PATH
                        Input .pb file path (.pb)
  --inputs INPUTS       (e.g.1) input,input_1,input_2 / (e.g.2)
                        images,input,param
  --outputs OUTPUTS     (e.g.1) output,output_1,output_2 / (e.g.2)
                        Identity,Identity_1,output
  --model_output_path MODEL_OUTPUT_PATH
                        The output folder path of the converted model file
```
```bash
usage: ir_weight_extractor.py [-h] -m MODEL -o OUTPUT_PATH

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
### 6-4. Converts saved_model to OpenVINO IR
```
$ python3 ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo_tf.py \
  --saved_model_dir saved_model \
  --output_dir openvino/reverse
```
### 6-5. Checking the structure of saved_model
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
