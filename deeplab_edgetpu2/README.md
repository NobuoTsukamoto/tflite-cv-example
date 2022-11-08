# DeepLab V3+ EdgeTPUV2 and AutoSeg EdgeTPU 

# Models
- [EdgeTPU-optimized Vision Models - Semantic segmentation task](https://github.com/tensorflow/models/tree/master/official/projects/edgetpu/vision#semantic-segmentation-task)
- [autoseg-edgetpu](https://tfhub.dev/google/collections/autoseg-edgetpu/1)
- [deeplab-edgetpu](https://tfhub.dev/google/collections/deeplab-edgetpu/1)

# Environment
- Edge TPU Compiler version 16.0.384591198
- Google Coral EdgeTPU (run EdgeTPU Model)
- TensorFlow Lite (v2.5 or higher)
- OpenCV (3.x or 4.x)

# Python Example.

Video capture
```
python3 deeplab_tflite_capture_opencv.py --help
usage: deeplab_tflite_capture_opencv.py [-h] --model MODEL [--width WIDTH] [--height HEIGHT] [--thread THREAD] [--videopath VIDEOPATH] [--output OUTPUT]

options:
  -h, --help            show this help message and exit
  --model MODEL         File path of Tflite model.
  --width WIDTH         Resolution width.
  --height HEIGHT       Resolution height.
  --thread THREAD       Num threads.
  --videopath VIDEOPATH
                        File path of Videofile.
  --output OUTPUT       File path of result.
```

Image
```
python3 deeplab_tflite_image_opencv.py --help
usage: deeplab_tflite_image_opencv.py [-h] --model MODEL [--input_shape INPUT_SHAPE] [--thread THREAD] [--input INPUT] [--output OUTPUT]

options:
  -h, --help            show this help message and exit
  --model MODEL         File path of Tflite model.
  --input_shape INPUT_SHAPE
                        Specify an input shape for inference.
  --thread THREAD       Num threads.
  --input INPUT         File path of image.
  --output OUTPUT       File path of result.
```

# YouTube video Link

## Coral Dev Board EdgeTPU AutosegEdgeTPU-XS with fusing argmax
[![](https://img.youtube.com/vi/2ywjDXRT6qo/0.jpg)](https://www.youtube.com/watch?v=2ywjDXRT6qo)

## Coral Dev Board EdgeTPU AutosegEdgeTPU-S with fusing argmax

[![](https://img.youtube.com/vi/-F9R51vFOS8/0.jpg)](https://www.youtube.com/watch?v=-F9R51vFOS8)
