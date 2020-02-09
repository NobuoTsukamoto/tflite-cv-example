# Edge TPU Object detection sample.

## Python examples
- [object_detection_capture_picamera.py](object_detection_capture_picamera.py)<br>
Raspberry Pi + PiCamera
- [object_detection_capture_opencv.py](object_detection_capture_opencv.py)<br>
OpenCV, VideoCapture or UVC
 Camera

## Model

### SSDLite MobileNet EdgeTPU Coco
- [Edge TPU Model](models/ssd_mobilenet_edgetpu_coco_edgetpu.tflite)
- [Original Model(Tensorflow detection model zoo)](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#pixel4-edge-tpu-models)
- [Label File](models/coco_labels.txt)
- Note: Be careful with the Edge TPU compiler and library versions.<br>
If the compiler version is 2.0.291256449, generate a model from [export_tflite_ssd_graph.py](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md). If you compile tflite in the pre-trained model of model zoo, it will not work properly.

## Usage
- object_detection_capture_picamera.py:<br>
``` 
    $ python3 ./object_detection_capture_picamera.py \
        --model=<PATH_TO_edgetpu.tflite> \
        --label=<PATH_TO_LABELS_TXT>
```
- object_detection_capture_opencv.py(Video file):<br>
```
    $ python3 object_detection_capture_opencv.py \
    --model=<PATH_TO_edgetpu.tflite> \ 
    --label=<PATH_TO_LABELS_TXT> \
    --videopath=<PATH_TO_VIDEO_FILE>
```
- object_detection_capture_opencv.py(UVC Camera):<br>
```
    # Note: To open camera using default backend just pass 0.
    $ python3 object_detection_capture_opencv.py \
    --model=<PATH_TO_edgetpu.tflite> \
    --label=<PATH_TO_LABELS_TXT> 
```