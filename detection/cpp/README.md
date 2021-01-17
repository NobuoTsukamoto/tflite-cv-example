# C++ CMake Examples.

## How to build.
### Install dependency packages.
```
$ sudo apt install libopencv-dev cmake libboost-dev
```

### Clone repository and init submodule.
```
$ git clone https://github.com/NobuoTsukamoto/edge_tpu.git
$ cd edge_tpu
$ git submodule init && git submodule update
```

### Build TensorFlow-Lite library.
```
$ ./tensorflow/tensorflow/lite/tools/make/download_dependencies.sh

# For arm7vl (Raspberry Pi)
$ ./tensorflow/tensorflow/lite/tools/make/build_rpi_lib.sh

# For aarch64 (Jetson Nano or Raspberry Pi OS 64bit
)
$ ./tensorflow/tensorflow/lite/tools/make/build_aarch64_lib.sh
```


### Build module.
```
$ cd detection/cpp/
$ mkdir build && cd build
$ cmake ..
$ make
```

## Usage
```
$ ./tflite_detection /home/pi/edge_tpu/detection/models/ssd_mobilenet_edgetpu_coco_edgetpu.tflite --label=/home/pi
/edge_tpu/detection/models/coco_labels.txt

# Full options.
$ ./tflite_detection --help
Usage: tflite_detection [params] input

        -?, -h, --help, --usage (value:true)
                show help command.
        -l, --label (value:.)
                path to label file.
        -n, --thread (value:2)
                num of thread to set tf-lite interpreter.
        -s, --score (value:0.5)
                score threshold.

        input
                path to tf-lite model file.
```
