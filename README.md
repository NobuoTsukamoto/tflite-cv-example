# TensorFlow Lite samples.

## About
TensorFlow Lite samples (Python/C++, Raspberry Pi/VisionFive 2/Windows/Linux).
 - CPU(XNNPACK) inference
 - Coral Edge TPU Delegate
 - GPU Delegate 

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
| [FFNet ](ffnet) | C++ | VisionFive 2 TensorFlow Lite GPU Delegate FFNet | TF-Lite<br>GPU delegate | Linux |


## Images

|Object detection|Camouflage|DeepLab|
|:--:|:--:|:--:|
|![detection](detection/g3doc/img/output.gif)|![camouflage](camouflage/g3doc/img/output.gif)|![deeplab](deeplab/g3doc/img/output.gif)|


|Segmentation|CenterNet|YOLOX|
|:--:|:--:|:--:|
|![segmentation](segmentation/g3doc/segmentation.gif)|![centernet](centernet/g3doc/img/centernet.gif)|![yolox](yolox/g3doc/yolox.gif)|


|DeepLab V3+ EdgeTPUV2 and AutoSeg EdgeTPU| VisionFive 2 TensorFlow Lite GPU Delegate<br>FFNet46NS CCC Mobile Pre-Down Fused-Argmax | VisionFive 2 TensorFlow Lite GPU Delegate<br>EfficientDet-Lite0 |
|:--:|:--:|:--:|
|YouTube Link<br>[![](https://img.youtube.com/vi/-F9R51vFOS8/mqdefault.jpg)](https://www.youtube.com/watch?v=-F9R51vFOS8)|YouTube Link<br>[![](https://img.youtube.com/vi/QDNdEaW8Z8U/mqdefault.jpg)](https://www.youtube.com/watch?v=QDNdEaW8Z8U)|YouTube Link<br>[![](https://img.youtube.com/vi/1SAccRvKuFM/mqdefault.jpg)](https://www.youtube.com/watch?v=1SAccRvKuFM)|

## Environment
- Coral Edge TPU USB Accelerator
- Raspberry Pi (3 B+ / 4) + PiCamera or UVC Camera
- Dev Board
- VisionFive 2
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

