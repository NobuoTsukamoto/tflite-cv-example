# Python Examples

## PiCamera Semantic Segmentation Example
Run Pi Camera stream.

### Usaege
``` $ python3 ./deeplab_picamera.py --model=<PATH_TO_MODEL_FILE>```<br>

- q key: End app.

 ### Option
- width:  Width of the frames in the camera stream.
- height: Height of the frames int the camera stream.

## Image Semantic Segmentation Example
Run single image. The segmentation image is saved to "save.png".

### Usaege
``` $ python3 ./deeplab_image.py --model=<PATH_TO_MODEL_FILE> --image=<PATH_TO_IMAGE_FILE>```<br>

## With Jetson Nano + PiCamera
Opencv VideoCapture With Gstreamer V4L2.<br>
It works around 4 FPS (deeplabv3_mnv2_dm05_pascal_trainaug_edgetpu.tflite).

### Usaege
``` $ python3 ./deeplab_videocapture.py --model=<PATH_TO_MODEL_FILE> --nano```<br>
