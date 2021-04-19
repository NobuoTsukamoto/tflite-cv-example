# CenterNet with TensorFlow Lite Python

## Usage
### Object Detection
```
$ python3 centernet_tflite_capture_opencv.py \
    --model ../models/centernet_mobilenetv2_fpn_od.tflite \
    --label ../models/coco_labels.txt
```
### Keypoint detection
```
$ python3 centernet_tflite_capture_opencv.py \
    --model ../models/centernet_mobilenetv2_fpn_kpts.tflite \
    --label ../models/coco_labels.txt --keypoint
```
