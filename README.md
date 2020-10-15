# openvino2tensorflow
This script converts the OpenVINO IR model to Tensorflow's saved_model, tflite, h5 and pb.

## 1. Environment
- TensorFlow v2.3.1
- OpenVINO_2021.1.110

## 2. Supported Layers
|No.|OpenVINO Layer|TF Layer|Remarks|
|:--:|:--|:--|:--|
|1|Parameter|Input|Input|
|2|Const|Const||
|3|Convolution|Conv2D||
|4|Add|Add||
|5|ReLU|ReLU||
|6|MaxPool|max_pool||
|7|GroupConvolution|DepthwiseConv2D||
|8|ConvolutionBackpropData|Conv2DTranspose||
|9|Concat|Concat||
|10|Multiply|Multiply||
|11|Tanh|Tanh||
|12|Result|Identity|Output|

## 3. Execution sample
```bash
$ python3 openvino2tensorflow.py \
  --model_path=openvino/448x448/FP32/Resnet34_3inputs_448x448_20200609.xml \
  --output_saved_model=True \
  --output_pb=True \
  --output_weight_quant_tflite=True \
  --output_float16_quant_tflite=True \
  --output_no_quant_float32_tflite=True
```

## 4. Output image
![Screenshot 2020-10-16 00:08:40](https://user-images.githubusercontent.com/33194443/96149093-e38fa700-0f43-11eb-8101-65fc20b2cc8f.png)
