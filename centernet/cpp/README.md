# CenterNet with TensorFlow Lite C++ XNNPACK delegate
Please note that this sample is a Keypoint detection model only.

## Models
- [tensorflow/models](https://github.com/tensorflow/models)
- [CenterNet on-device with TensorFlow Lite Colab](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/centernet_on_device.ipynb)

## How to build

### Install dependency
Install the required software.
```
# Ubuntu or Raspberry Pi
$ sudo apt install build-essential cmake pkg-config
$ sudo apt install libopencv-dev libboost-all-dev

# Fedora
$ sudo dnf cmake
$ sudo dnf install boost-devel
```

Install bazel. Here is an example of Raspberry Pi 64bit OS.
```
$ wget https://github.com/bazelbuild/bazel/releases/download/4.0.0/bazel-4.0.0-linux-arm64
$ sudo mv bazel-4.0.0-linux-arm64 /usr/local/bin/bazel
$ sudo chmod +x /usr/local/bin/bazel
```

### Clone repository and init submodule.
```
$ cd ~
$ git clone https://github.com/NobuoTsukamoto/edge_tpu.git
$ cd edge_tpu
$ git submodule init && git submodule update
```

### Build TensorFlow-Lite library.
```
$ cd tensorflow
$ bazel build  --define tflite_with_xnnpack=true -c opt //tensorflow/lite:libtensorflowlite.so
```
After the build is complete, `libtensorflowlite.so` will be created.
```
$ ls tensorflow/bazel-bin/tensorflow/lite/libtensorflowlite.so
tensorflow/bazel-bin/tensorflow/lite/libtensorflowlite.so
```
Download the header files needed for the build.
```
$ ./tensorflow/lite/tools/make/download_dependencies.sh
```
### Build module.
```
$ cd ../centernet/cpp/
$ mkdir build && cd build
$ cmake ..
$ make
```

## Usage
Camera input.
```
./centernet_detection \
  /home/pi/edge_tpu/centernet/models/centernet_mobilenetv2_fpn_kpts.tflite \
  --label=/home/pi/edge_tpu/centernet/models/coco_labels.txt
```
Video file input.
```
./centernet_detection \
  /home/pi/edge_tpu/centernet/models/centernet_mobilenetv2_fpn_kpts.tflite \
  --label=/home/pi/edge_tpu/centernet/models/coco_labels.txt
  --videopath=_PATH_TO_VIDEO_FILE_PATH_
```
Full option.
```
./centernet_detection --help
Usage: centernet_detection [params] input 

        -?, -h, --help, --usage (value:true)
                show help command.
        -H, --height (value:480)
                camera resolution height.
        -W, --width (value:640)
                camera resolution width.
        -l, --label
                path to label file.
        -n, --thread (value:2)
                number of threads specified in XNNPackDelegateOptions.
        -o, --output
                file path of output videofile.
        -s, --score (value:0.5)
                score threshold.
        -v, --videopath
                file path of videofile.

        input
                path to centernet keypoint tf-lite model file.
```