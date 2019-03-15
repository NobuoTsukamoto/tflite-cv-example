# Edge TPU samples.

## About
Edge tpu python sample.
 - Object detection by PiCamera (Raspberry Pi).
 - Image classifilcation by PiCamera (Raspberry Pi).

## Environment
- Coral USB Accelerator
- Raspberry Pi (3 B+)
- PiCamera

## Installation
- OpenCV (3.4.5 or higher)
- Edge TPU Python library [(Get started with the USB Accelerator)](https://coral.withgoogle.com/tutorials/accelerator/)

## Usage
Object detection:<br>
``` $ python3 ./object_detection_capture_picamera.py --model=<PATH_TO_edgetpu.tflite> --label=<PATH_TO_LABELS_TXT>```

Image classification:<br>
``` $ python3 ./object_detection_capture_picamera.py --model=<PATH_TO_edgetpu.tflite> --label=<PATH_TO_LABELS_TXT>```

Full command-line options:<br>
``` $ python3 ./xxxx.py -h```

## Reference
- [Get started with the USB Accelerator](https://coral.withgoogle.com/tutorials/accelerator/)
- [TensorFlow models on the Edge TPU](https://coral.withgoogle.com/tutorials/edgetpu-models-intro/#model-requirements)
- [Models Built for the Edge TPU](https://coral.withgoogle.com/models/)
