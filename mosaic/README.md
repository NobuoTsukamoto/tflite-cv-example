# MOSAIC: Mobile Segmentation via decoding Aggregated Information and encoded Context

# Models
- [TensorFlow Official Models - MOSAIC: Mobile Segmentation via decoding Aggregated Information and encoded Context](https://github.com/tensorflow/models/tree/master/official/projects/mosaic)

# Environment
- TensorFlow Lite (v2.10 or higher)
- OpenCV (3.x or 4.x)

# Export TF-Lite with argmax / fused argmax
- https://github.com/NobuoTsukamoto/models/tree/master/official/projects/mosaic/serving

# Python Example.

Video capture
```
python3 mosaic_tflite_capture_opencv.py --help
usage: mosaic_tflite_capture_opencv.py [-h] --model MODEL [--thread THREAD] [--videopath VIDEOPATH] [--output OUTPUT]

options:
  -h, --help            show this help message and exit
  --model MODEL         File path of Tflite model.
  --thread THREAD       Num threads.
  --videopath VIDEOPATH
                        File path of Videofile.
  --output OUTPUT       File path of result.
```

Image
```
python3 mosaic_tflite_image_opencv.py --help
usage: mosaic_tflite_image_opencv.py [-h] --model MODEL [--thread THREAD] [--input INPUT] [--output OUTPUT]

options:
  -h, --help       show this help message and exit
  --model MODEL    File path of Tflite model.
  --thread THREAD  Num threads.
  --input INPUT    File path of image.
  --output OUTPUT  File path of result.
```


