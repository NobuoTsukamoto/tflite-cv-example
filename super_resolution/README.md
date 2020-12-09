# Super resolution using ESRGAN
Based on the TensorFlow tutorial, it is an image super resolution model that runs on TensorFlow Lite.<br>
Only the TensorFlow Lite(FP32, FP16, INT8) model.

## Models
- [Super resolution with TensorFlow Lite](https://github.com/tensorflow/examples/blob/master/lite/examples/super_resolution/ml/super_resolution.ipynb)
- [Super resolution with Post-training integer quantization with int16 activations](https://gist.github.com/NobuoTsukamoto/7102843b00b0b3d5b11d2e477315d54d)

## Usage
- image_segmentation_tflite_capture_opencv.py:<br>
``` 
    # move to edge_tpu/super_resolution

    $ $ python esrgan_capture_opencv.py --model model/ESRGAN.tflite --thread 4
```