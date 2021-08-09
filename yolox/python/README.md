# YOLOX with TensorFlowLite - Python

## Description
This sample contains Python code that running YOLOX-TensorFlow Lite model.

## Reference
- [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
  - [YOLOX-ONNXRuntime in Python](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime)  
  I referred to the implementation of the inference code.
- [PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
  - [YOLOX models](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/132_YOLOX)

## Environment
- HW
  - PC or Raspberry Pi
  - Camera (optional)
- SW
  - TensorFlow Lite v2.5
  - OpenCV v4.5

## Samples

|Name | Description |
|:--  |:--          |
| [yolox_tflite_demo.py](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/132_YOLOX/demo/tflite/yolox_tflite_demo.py) | Camera or Video file input demo.<br>This sample is available at [PINTO030/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/132_YOLOX/demo/tflite).  |
| [yolox_tflite_coco_metrics.py](./yolox_tflite_coco_metrics.py) |Benchmark COCO metrics with YOLOX.|
| [yolox_tflite_image.py](./yolox_tflite_image.py) | Image file input demo. |


## How to

Clone this repository.
```
cd ~
git clone https://github.com/NobuoTsukamoto/tflite-cv-example.git
cd tflite-cv-example
git submodule init && git submodule update
```

Clone [PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo) and download YOLOX models.
```
cd ~
git clone https://github.com/PINTO0309/PINTO_model_zoo.git
cd PINTO_model_zoo/132_YOLOX/

# YOLOX Nano
download_nano.sh

# YOLOX Tiny
download_tiny.sh

```
Finally, Run demo.


## LICENSE

The following files are licensed under [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).

- [yolox/python/utils](yolox/python/utils)