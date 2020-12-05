# Usage
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