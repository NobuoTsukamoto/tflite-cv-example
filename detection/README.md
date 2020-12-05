# TensorFlow Lite / Edge TPU Object detection samples.

![Image](g3doc/img/output.gif)

## List of samples

| Name | Language | Description | Backend | OS |
|:---|:---|:---|:---|:---|
|[Benchmark](python/object_detection_benchmark_tflite_opencv.py) | Python | TensorFlow Lite Object detection benchmark with OpenCV. | TensorFLow Lite (EdgeTPU delegate) | Windows, Linux |
|[Capture using PyCoral](python/object_detection_capture_opencv.py) | Python | Object detection for camera or video capture using OpenCV and PyCoral API. | PyCoral | Windows, Linux |
|[PiCamera using PyCoral](python/object_detection_capture_picamera.py) | Python | Object detection for camera capture using PiCamera and PyCoral API. | PyCoral | Raspberry Pi |
|[Capture using TensorFlow Lite](python/object_detection_tflite_capture_opencv.py)| Python | Object detection for camera or video capture using OpenCV and TensorFlow Lite API. | TensorFlow Lite (EdgeTPU delegate) | Windows, Linux |
|[C++ CMake](cpp)| C++ | C++ CMake Project.<br>Object detection for camera or video capture using OpenCV and TensorFlow Lite API. | TensorFlow Lite (EdgeTPU delegate) | Windows, Linux |
|[VC++](vc_tflite) | Visual C++ | Visual Studio 2019 (VC++) Sample.<br>TensorFlow Lite and TensorFlow Lite EdgeTPU delegate with Visual Studio. | TensorFlow Lite (EdgeTPU delegate) | Windows |
<br>

## Models


## Coral Edge TPU models
[Coral Pre-compiled models](https://coral.ai/models/)

## TensorFlow 1 Detection Model Zoo
[This notebook](https://gist.github.com/NobuoTsukamoto/832905aa765f6faa16f53d6dddf61bd2) converts the pre-trained model of "TensorFlow 1 Detection Model Zoo" into TF-Lite or Edge TPU Model.

|Model Name|Output model type|
|:---|:---|
|ssd_mobilenet_v2_coco|INT8, EdgeTPU|
|ssd_mobilenet_v3_large_coco|FP32|
|ssd_mobilenet_v3_small_coco|FP32|
|ssd_mobilenet_v2_mnasfpn_coco|FP32|
|ssdlite_mobiledet_cpu_coco|FP32|
|ssdlite_mobiledet_edgetpu_coco|INT8, EdgeTPU|
|ssd_mobilenet_edgetpu_coco|INT8, EdgeTPU|
