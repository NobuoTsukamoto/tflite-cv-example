# Edge TPU Object detection sample.

## Python examples
- [object_detection_capture_picamera.py](object_detection_capture_picamera.py)<br>
Raspberry Pi + PiCamera
- [object_detection_capture_opencv.py](object_detection_capture_opencv.py)<br>
OpenCV, VideoCapture or UVC
 Camera

## Model

### SSDLite MobileNet EdgeTPU Coco
- [Edge TPU Model](models/ssd_mobilenet_edgetpu_coco_edgetpu.tflite)
- [Original Model(Tensorflow detection model zoo)](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#pixel4-edge-tpu-models)
- [Label File](models/coco_labels.txt)
- Note: Be careful with the Edge TPU compiler and library versions.<br>
If the compiler version is 2.0.291256449, generate a model from [export_tflite_ssd_graph.py](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md). If you compile tflite in the pre-trained model of model zoo, it will not work properly.<br>
Example.
    - [Install Tensorflow Object Detection API.](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
    - Download models and unzip files.
```
    # From tensorflow/models/research/
    $ python object_detection/export_tflite_ssd_graph.py \
        --pipeline_config_path=ssdlite_mobilenet_edgetpu_coco_quant/pipeline.config \
        --trained_checkpoint_prefix=./ssdlite_mobilenet_edgetpu_coco_quant/model.ckpt \
        --output_directory=ssdlite_mobilenet_edgetpu_coco_quant \
        --add_postprocessing_op=true
    $ tflite_convert \
        --output_file=ssdlite_mobilenet_edgetpu_coco_quant/output_tflite_graph.tflite \
        --graph_def_file=ssdlite_mobilenet_edgetpu_coco_quant/tflite_graph.pb \
        --inference_type=QUANTIZED_UINT8 \
        --input_arrays=normalized_input_image_tensor \
        --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 \
        --mean_values=128 \
        --std_dev_values=128 \
        --input_shapes=1,320,320,3 \
        --change_concat_input_ranges=false \
        --allow_nudging_weights_to_use_fast_gemm_kernel=true 
        --allow_custom_ops
    $ cd ssdlite_mobilenet_edgetpu_coco_quant
    $ edgetpu_compiler -s output_tflite_graph.tflite 

```


## Usage
- object_detection_capture_picamera.py:<br>
``` 
    $ python ./object_detection_capture_picamera.py \
        --model=<PATH_TO_edgetpu.tflite> \
        --label=<PATH_TO_LABELS_TXT>
```
- object_detection_capture_opencv.py(Video file):<br>
```
    $ python object_detection_capture_opencv.py \
    --model=<PATH_TO_edgetpu.tflite> \ 
    --label=<PATH_TO_LABELS_TXT> \
    --videopath=<PATH_TO_VIDEO_FILE>
```
- object_detection_capture_opencv.py(UVC Camera):<br>
```
    # Note: To open camera using default backend just pass 0.
    $ python object_detection_capture_opencv.py \
    --model=<PATH_TO_edgetpu.tflite> \
    --label=<PATH_TO_LABELS_TXT> 
```