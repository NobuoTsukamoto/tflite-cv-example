# Edge TPU Semantic Segmentation using DeepLabv3

![Image](g3doc/img/output.gif)

# Models
The official ["Edge TPU model"](https://coral.withgoogle.com/models/) has been released.<br>
If you want to create your own model, see [this link (Quantize DeepLab model for faster on-device inference)](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/quantize.md).


# Note
Be careful with the compiler and library versions. Previous versions do not work properly.
- Edge TPU Compiler version 2.0.258810407 or hihger
- TPU Python library virsion 2.11.1

# Limitations
- Labels only Pascal VOC label format.

# Python Example.
[See this](./python/README.md)

# C++ Example.
[See this](./cpp/README.md)
