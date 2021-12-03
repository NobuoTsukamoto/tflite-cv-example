# TensorFlow Lite, Coral Edge TPU samples.

## About
TensorFlow Lite, Coral Edge TPU samples (Python/C++, Raspberry Pi/Windows/Linux).

## List of samples.

| Name | Language | Description | API | OS |
|:---|:---|:---|:---|:---|
|[Camouflage](camouflage)| Python | Object detection and camouflage objects by PiCamera. | PyCoral | Linux<br>Windows |
|[Classify](classify) | Python | Image classifilcation by PiCamera or Video Capture.| TF-Lite<br>PyCoral | Linux<br>Windows |
|[CenterNet](centernet)|Python<br>C++|CenterNet on-device with TensorFlow Lite.|TF-Lite|Liux<br>Windows|
| [DeepLab](deeplab) | Python<br>C++ | Semantic Segmentation using DeepLab v3. | TF-Lite<BR>EdgeTPU API | Linux<br>Windows |
| [Object detection](detection) | Python<br>C++<br>VC++ | Object detection by PiCamera or Video Capture. | TF-Lite<br>PyCoral | Linux<br>Windows |
| [U-Net MobileNet v2](segmentation) | Python | Image segmentation model U-Net MobileNet v2. | TF-Lite | Linux<br>Windows 
| [Super resolution](super_resolution) | Python | Super resolution using ESRGAN. | TF-Lite | Linux<br>Windows |
| [YOLOX](yolox/python) | Python | YOLOX with TensorFlow Lite. | TF-Lite | Linux<br>Windows |
| [DeepLab V3+ EdgeTPUV2 and AutoSeg EdgeTPU](deeplab_edgetpu2) | Python | DeepLab V3+ EdgeTPUV2 and AutoSeg EdgeTPU with TensorFlow Lite. | TF-Lite<br>EdgeTPU | Linux<br>Windows |


## Images

|Object detection|Camouflage|DeepLab|
|:--:|:--:|:--:|
|![detection](detection/g3doc/img/output.gif)|![camouflage](camouflage/g3doc/img/output.gif)|![deeplab](deeplab/g3doc/img/output.gif)|


|Segmentation|Camouflage|YOLOX|
|:--:|:--:|:--:|
|![segmentation](segmentation/g3doc/segmentation.gif)|![centernet](centernet/g3doc/img/centernet.gif)|![yolox](yolox/g3doc/yolox.gif)|


|DeepLab V3+ EdgeTPUV2 and AutoSeg EdgeTPU|
|:--:|
|YouTube Link<br>[![](https://img.youtube.com/vi/-F9R51vFOS8/mqdefault.jpg)](https://www.youtube.com/watch?v=-F9R51vFOS8)|

## Environment
- Coral Edge TPU USB Accelerator
- Raspberry Pi (3 B+ / 4) + PiCamera or UVC Camera
- Dev Board
- x64 PC(Windows or Linux) + Video file or UVC Camera
- Python3

## Installation
- OpenCV with OpenCV's extra modules(3.4.5 or higher)
- TensorFlow Lite Runtime [(Python quickstart)](https://www.tensorflow.org/lite/guide/python).
- Edge TPU Python library [(Get started with the USB Accelerator)](https://coral.withgoogle.com/tutorials/accelerator/).

## Reference
- [Get started with the USB Accelerator](https://coral.withgoogle.com/tutorials/accelerator/)
- [TensorFlow models on the Edge TPU](https://coral.withgoogle.com/tutorials/edgetpu-models-intro/#model-requirements)
- [Models Built for the Edge TPU](https://coral.withgoogle.com/models/)
- [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)

