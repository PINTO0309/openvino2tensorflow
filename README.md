# openvino2tensorflow
This script converts the OpenVINO IR model to Tensorflow's saved_model, tflite, h5 and pb.

**I'm continuing to add more layers of support and bug fixes on a daily basis. If you have a model that you are having trouble converting, please share the `.bin` and `.xml` with the issue. I will try to convert as much as possible.**

## 1. Environment
- TensorFlow v2.3.1
- OpenVINO 2021.1.110

## 2. Use case

- PyTorch (NCHW) -> ONNX (NCHW) -> OpenVINO (NCHW) ->
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TFLite (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TFJS (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> TF-TRT (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> EdgeTPU (NHWC)
  - -> **`openvino2tensorflow`** -> Tensorflow/Keras (NHWC) -> CoreML (NHWC)

## 3. Supported Layers
Currently, only 4D tensors are supported as input tensors.
|No.|OpenVINO Layer|TF Layer|Remarks|
|:--:|:--|:--|:--|
|1|Parameter|Input|Input (4D tensor only)|
|2|Const|Constant, Bias||
|3|Convolution|Conv2D||
|4|Add|Add||
|5|ReLU|ReLU||
|6|PReLU|PReLU||
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
|19|Swish|Swish||
|20|Interpolate|ResizeNearestNeighbor, ResizeBilinear||
|21|ShapeOf|Shape||
|22|Convert|Cast||
|23|StridedSlice|Strided_Slice|WIP|
|24|Pad|Pad, MirrorPad||
|25|Clamp|ReLU6, Clip||
|26|TopK|ArgMax, top_k||
|27|Transpose|Transpose||
|28|Squeeze|Squeeze||
|29|ReduceMean|reduce_mean||
|30|ReduceMax|reduce_max||
|31|ReduceMin|reduce_min||
|32|ReduceSum|reduce_sum||
|33|ReduceProd|reduce_prod||
|34|Subtract|Subtract||
|35|MatMul|MatMul||
|36|Reshape|Reshape||
|37|Range|Range|WIP|
|38|Exp|Exp||
|39|Abs|Abs||
|40|SoftMax|SoftMax||
|41|Negative|Negative||
|42|Maximum|Maximum|No broadcast|
|43|Minimum|Minimum|No broadcast|
|44|Acos|Acos||
|45|Acosh|Acosh||
|46|Asin|Asin||
|47|Asinh|Asinh||
|48|Atan|Atan||
|49|Atanh|Atanh||
|50|Ceiling|Ceil||
|51|Cos|Cos||
|52|Cosh|Cosh||
|53|Gather|Gather||
|54|Divide|divide_no_nan||
|55|Erf|Erf||
|56|Floor|Floor||
|57|FloorMod|FloorMod||
|58|HSwish|HardSwish|x\*ReLU6(x+3)\*0.16666667|
|59|Log|Log||
|60|Power|Pow|No broadcast|
|61|Mish|Mish|x\*Tanh(softplus(x))|
|62|Selu|Selu||
|63|Result|Identity|Output|

## 4 Setup

```bash
pip install git+https://github.com/PINTO0309/openvino2tensorflow --upgrade
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
  --debug               debug mode switch
  --debug_layer_number DEBUG_LAYER_NUMBER
                        The last layer number to output when debugging. Used
                        only when --debug=True.
```
```bash
usage: pb_to_saved_model [-h] --pb_file_path PB_FILE_PATH --inputs INPUTS
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
### 6-3. Converts saved_model to OpenVINO IR
```
$ python3 ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo_tf.py \
  --saved_model_dir saved_model \
  --output_dir openvino/reverse
```
### 6-4. Checking the structure of saved_model
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
