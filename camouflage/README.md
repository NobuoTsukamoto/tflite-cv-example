# Edge TPU Camouflage PyCamera

![Image](g3doc/img/output.gif)

## Download model
Using the Edge TPU pre-trained model.<br>
Download SSD MoblieNet v1 coco or SSD MoblieNet v2 coco model.<br>
See [Coral Models](https://coral.withgoogle.com/models/).

## Usage
``` $ python3 ./camouflage_picamera.py --model=<PATH_TO_edgetpu.tflite> --label=./label.txt```

 - Space key: toggle detection and camouflage.
 - q key: End app.

 ### Option
 - top_k: The maximum number of detected objects to return (DetectWithImage).
 - threshold: Minimum confidence threshold for detected objects (DetectWithImage).
 - width:  Width of the frames in the camera stream.
 - height: Height of the frames int the camera stream.

## If you want to change the object you want to camouflage
You can change the target object by changing the label.txt file.<br>
Default is banana(40 banana). <br>
Select the target object from the coco label(index and name). Or use the model you created.
