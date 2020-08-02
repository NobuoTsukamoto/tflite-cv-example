# Edge TPU Object detection sample.

![Image](g3doc/img/output.gif)

## Python examples
- Using Edge TPU Python API (Edge TPU Model)
    - [object_detection_capture_picamera.py](object_detection_capture_picamera.py)<br>
    Raspberry Pi + PiCamera
    - [object_detection_capture_opencv.py](object_detection_capture_opencv.py)<br>
    OpenCV, VideoCapture or UVC Camera
- Using TensorFlow Lite interpreter (TF-Lite Model, Edge TPU Model)
    - [object_detection_benchmark_tflite_opencv.py](object_detection_benchmark_tflite_opencv.py)<br>
    Benchmark script
    - [object_detection_tflite_capture_opencv.py](object_detection_tflite_capture_opencv.py)<br>
    OpenCV, VideoCapture or UVC Camera

## Models

Pre-trained models.
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



## Usage
- object_detection_capture_picamera.py:<br>
``` 
    $ python ./object_detection_capture_picamera.py \
        --model=<PATH_TO_edgetpu.tflite> \
        --label=<PATH_TO_LABELS_TXT>
```
- object_detection_capture_opencv.py(Video file):<br>
```
    $ python object_detection_capture_opencv.py \
    --model=<PATH_TO_edgetpu.tflite> \ 
    --label=<PATH_TO_LABELS_TXT> \
    --videopath=<PATH_TO_VIDEO_FILE>
```
- object_detection_capture_opencv.py(UVC Camera):<br>
```
    # Note: To open camera using default backend just pass 0.
    $ python object_detection_capture_opencv.py \
    --model=<PATH_TO_edgetpu.tflite> \
    --label=<PATH_TO_LABELS_TXT> 
```