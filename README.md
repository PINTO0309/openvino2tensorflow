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
|17|Swish|Swish||
|18|Interpolate|ResizeNearestNeighbor, ResizeBilinear||
|19|ShapeOf|Shape||
|20|Convert|Cast||
|21|StridedSlice|Strided_Slice|WIP|
|22|Pad|Pad, MirrorPad||
|23|Clamp|ReLU6, Clip||
|24|TopK|ArgMax, top_k||
|25|Transpose|Transpose||
|26|Squeeze|Squeeze||
|27|ReduceMean|reduce_mean||
|28|ReduceMax|reduce_max||
|29|ReduceMin|reduce_min||
|30|ReduceSum|reduce_sum||
|31|ReduceProd|reduce_prod||
|32|Subtract|Subtract||
|33|MatMul|MatMul||
|34|Reshape|Reshape||
|35|Range|Range|WIP|
|36|Exp|Exp||
|37|Abs|Abs||
|38|SoftMax|SoftMax||
|39|Negative|Negative||
|40|Maximum|Maximum|No broadcast|
|41|Minimum|Minimum|No broadcast|
|42|Acos|Acos||
|43|Acosh|Acosh||
|44|Asin|Asin||
|45|Asinh|Asinh||
|46|Atan|Atan||
|47|Atanh|Atanh||
|48|Ceiling|Ceil||
|49|Cos|Cos||
|50|Cosh|Cosh||
|51|Gather|Gather||
|52|Divide|divide_no_nan||
|53|Erf|Erf||
|54|Floor|Floor||
|55|FloorMod|FloorMod||
|56|HSwish|HardSwish|x\*ReLU6(x+3)\*0.16666667|
|57|Log|Log||
|58|Power|Pow|No broadcast|
|59|Mish|Mish|x\*Tanh(softplus(x))|
|60|Selu|Selu||
|61|Result|Identity|Output|

## 4. Usage
```bash
usage: openvino2tensorflow.py [-h] --model_path MODEL_PATH
                              [--model_output_path MODEL_OUTPUT_PATH]
                              [--output_saved_model OUTPUT_SAVED_MODEL]
                              [--output_h5 OUTPUT_H5] [--output_pb OUTPUT_PB]
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
usage: pb_to_saved_model.py [-h] --pb_file_path PB_FILE_PATH --inputs INPUTS
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

## 5. Execution sample
### 5-1. Conversion of OpenVINO IR to Tensorflow models
OutOfMemory may occur when converting to saved_model or h5 when the file size of the original model is large, please try the conversion to a pb file alone.
```
$ python3 openvino2tensorflow.py \
  --model_path=openvino/448x448/FP32/Resnet34_3inputs_448x448_20200609.xml \
  --output_saved_model True \
  --output_pb True \
  --output_weight_quant_tflite True \
  --output_float16_quant_tflite True \
  --output_no_quant_float32_tflite True
```
### 5-2. Convert Protocol Buffer (.pb) to saved_model
This tool is useful if you want to check the internal structure of pb files, tflite files, .h5 files, coreml files and IR (.xml) files. **https://lutzroeder.github.io/netron/**
```
$ python3 pb_to_saved_model.py \
  --pb_file_path model_float32.pb \
  --inputs inputs:0 \
  --outputs Identity:0
```
### 5-3. Converts saved_model to OpenVINO IR
```
$ python3 ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo_tf.py \
  --saved_model_dir saved_model \
  --output_dir openvino/reverse
```
### 5-4. Checking the structure of saved_model
```
$ saved_model_cli show \
  --dir saved_model \
  --tag_set serve \
  --signature_def serving_default
```

## 6. Output sample
![Screenshot 2020-10-16 00:08:40](https://user-images.githubusercontent.com/33194443/96149093-e38fa700-0f43-11eb-8101-65fc20b2cc8f.png)


## 7. Model Structure
**[https://digital-standard.com/threedpose/models/Resnet34_3inputs_448x448_20200609.onnx](https://github.com/digital-standard/ThreeDPoseUnityBarracuda#download-and-put-files)**
|ONNX|OpenVINO|TFLite|
|:--:|:--:|:--:|
|![Resnet34_3inputs_448x448_20200609 onnx_](https://user-images.githubusercontent.com/33194443/96398683-62683680-1207-11eb-928d-e4cb6c8cc188.png)|![Resnet34_3inputs_448x448_20200609 xml](https://user-images.githubusercontent.com/33194443/96153010-23f12400-0f48-11eb-8186-4bbad73b517a.png)|![model_float32 tflite](https://user-images.githubusercontent.com/33194443/96153019-26ec1480-0f48-11eb-96be-0c405ee2cbf7.png)|
