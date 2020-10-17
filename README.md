# openvino2tensorflow
This script converts the OpenVINO IR model to Tensorflow's saved_model, tflite, h5 and pb.

## 1. Environment
- TensorFlow v2.3.1
- OpenVINO_2021.1.110

## 2. Supported Layers
|No.|OpenVINO Layer|TF Layer|Remarks|
|:--:|:--|:--|:--|
|1|Parameter|Input|Input|
|2|Const|Constant, Bias||
|3|Convolution|Conv2D||
|4|Add|Add||
|5|ReLU|ReLU||
|6|PReLU|PReLU||
|7|MaxPool|MaxPool2D||
|8|AvgPool|AveragePooling2D||
|9|GroupConvolution|DepthwiseConv2D||
|10|ConvolutionBackpropData|Conv2DTranspose||
|11|Concat|Concat||
|12|Multiply|Multiply||
|13|Tanh|Tanh||
|14|Elu|Elu||
|15|Sigmoid|Sigmoid||
|16|Interpolate|ResizeNearestNeighbor / ResizeBilinear||
|17|ShapeOf|Shape||
|18|Convert|Cast||
|19|StridedSlice|Strided_Slice||
|20|Pad|Pad, MirrorPad||
|21|Result|Identity|Output|

## 3. Usage
```bash
usage: openvino2tensorflow.py [-h] --model_path MODEL_PATH
                              [--model_output_path MODEL_OUTPUT_PATH]
                              [--output_saved_model OUTPUT_SAVED_MODEL]
                              [--output_h5 OUTPUT_H5] [--output_pb OUTPUT_PB]
                              [--output_no_quant_float32_tflite OUTPUT_NO_QUANT_FLOAT32_TFLITE]
                              [--output_weight_quant_tflite OUTPUT_WEIGHT_QUANT_TFLITE]
                              [--output_float16_quant_tflite OUTPUT_FLOAT16_QUANT_TFLITE]

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
```

## 4. Execution sample
```
$ python3 openvino2tensorflow.py \
  --model_path=openvino/448x448/FP32/Resnet34_3inputs_448x448_20200609.xml \
  --output_saved_model=True \
  --output_pb=True \
  --output_weight_quant_tflite=True \
  --output_float16_quant_tflite=True \
  --output_no_quant_float32_tflite=True
```

## 5. Output sample
![Screenshot 2020-10-16 00:08:40](https://user-images.githubusercontent.com/33194443/96149093-e38fa700-0f43-11eb-8101-65fc20b2cc8f.png)


## 6. Model Structure
|OpenVINO|TFLite|
|:--:|:--:|
|![Resnet34_3inputs_448x448_20200609 xml](https://user-images.githubusercontent.com/33194443/96153010-23f12400-0f48-11eb-8186-4bbad73b517a.png)|![model_float32 tflite](https://user-images.githubusercontent.com/33194443/96153019-26ec1480-0f48-11eb-96be-0c405ee2cbf7.png)|
