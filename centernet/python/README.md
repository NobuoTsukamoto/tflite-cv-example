# Models
- [tensorflow/models](https://github.com/tensorflow/models)
- [CenterNet on-device with TensorFlow Lite Colab](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/centernet_on_device.ipynb)

# Usage
```
$ cd edge_tpu/centernet/python

# detection
$ python3 centernet_tflite_capture_opencv.py --model ./models/centernet_mobilenetv2_fpn_od.tflite --label ./models/coco_labels.txt

# keypoint
$ python3 centernet_tflite_capture_opencv.py --model ./models/centernet_mobilenetv2_fpn_kpts.tflite --label ./models/coco_labels.txt --keypoint
```
