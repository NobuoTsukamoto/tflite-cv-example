# Edge TPU Semantic Segmentation using DeepLabv3

![Image](g3doc/img/output.gif)

# Models
Please check this [link (Quantize DeepLab model for faster on-device inference)](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/quantize.md) for details.<br>
There is an Edge TPU Model compiled from a pre-trained model. Models are in the **models** folder.

Edge TPU Model name                                 | TF-Lite Model(pre-trainded)
----------------------------------------------------|-------------------------------------------
deeplabv3_mnv2_dm05_pascal_trainaug_edgetpu.tflite  | [mobilenetv2_dm05_coco_voc_trainaug_8bit](http://download.tensorflow.org/models/deeplabv3_mnv2_dm05_pascal_train_aug_8bit_2019_04_26.tar.gz)
deeplabv3_mnv2_pascal_train_aug_edgetpu.tflite      | [mobilenetv2_coco_voc_trainaug_8bit](http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_8bit_2019_04_26.tar.gz)


# Note
Be careful with the compiler and library versions. Previous versions do not work properly.
- Edge TPU Compiler version 2.0.258810407
- TPU Python library virsion 2.11.1

# Limitations
- Labels only Pascal VOC label format.

# Python Example.
[See this](./python/README.md)

# C++ Example.
[See this](./cpp/README.md)
