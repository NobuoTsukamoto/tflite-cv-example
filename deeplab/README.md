# Edge TPU DeepLabv3

![Image](g3doc/img/output.gif)

# Models
Please check this [link(Quantize DeepLab model for faster on-device inference)](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/quantize.md) for details.<br>
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

# PiCamera Segmantation Example
Run Pi Camera stream.

## Usaege
``` $ python3 ./deeplab_picamera.py --model=./model/deeplabv3_mnv2_dm05_pascal_trainaug_edgetpu.tflite```<br>
or<br>
``` $ python3 ./deeplab_picamera.py --model=./model/deeplabv3_mnv2_pascal_train_aug_edgetpu.tflite```<br>

- q key: End app.

 ### Option
- width:  Width of the frames in the camera stream.
- height: Height of the frames int the camera stream.

# Image Segmantation Example
Run single image. The segmentation image is saved to "save.png".

## Usaege
``` $ python3 ./deeplab_image.py --model=./model/deeplabv3_mnv2_dm05_pascal_trainaug_edgetpu.tflite --image=<PATH_TO_IMAGE_FILE>```<br>
or<br>
``` $ python3 ./deeplab_image.py --model=./model/deeplabv3_mnv2_pascal_train_aug_edgetpu.tflite --image=<PATH_TO_IMAGE_FILE>```<br>
