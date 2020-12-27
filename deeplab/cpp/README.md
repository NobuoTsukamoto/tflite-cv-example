# C++ Examples.
This example performs segmentation with the TensorFlow Lite C++ API using the given deeplab v3 model, and OpenCV VideoCapture IF.

## Examples.
- [**Benchmark**](benchmark): Benchmark to measure inference time. [See blog for benchmark results](https://nextremer-nbo.blogspot.com/)
- [**Camouflage**](camouflage): Camouflage the specified label with a noise image. [watch this video.](https://www.youtube.com/watch?v=b46mX0C4Mqo)
- [**Segmentation**](segmentation): Draw the detected label with a color map. [watch this video.](https://youtube.com/watch?v=JtUR1ofaqN0)
- [**AfterImage**](motion): Draws the afterimage of the specified label. [watch this video.](https://www.youtube.com/watch?v=zQptVRlUwAM)
- [**Transparent**](transparent): The specified label becomes transparent.


## Reference
- [Google Coral Edge TPU with C++ on Jetson Nano](https://qiita.com/iwatake2222/items/3a09a2d26b022a5a8a95)
- [Build TensorFlow Lite for ARM64 boards](https://www.tensorflow.org/lite/guide/build_arm64)
- [Coral EdgeTPU C++ API overview](https://coral.withgoogle.com/docs/edgetpu/api-cpp/)
- [**How to create pre-built TensorflowLite library**](https://github.com/iwatake2222/EdgeTPU_CPP#how-to-create-pre-built-tensorflowlite-library)

# How to build.
This build method targets armv7l or aarch64.<br>

Install dependency packages.<br>
**Note (Raspberry Pi): For OpenCV, install the source build with GStreamer.**
```$ sudo apt-get install build-essential
$ sudo apt install -y curl wget cmake
$ sudo apt install -y libc6-dev libc++-dev libc++abi-dev
$ sudo apt install -y libusb-1.0-0
```

## Clone repository
```
$ git clone https://github.com/NobuoTsukamoto/edge_tpu.git
$ cd edge_tpu
$ git submodule init && git submodule update
```

## Build TensorFlow-Lite library
**Please refer to the following URL(github repository) for details of build TensorFlow Library. Thanks [@iwatake2222-san](https://twitter.com/iwatake2222)!**
- [**iwatake2222 / EdgeTPU_CPP**](https://github.com/iwatake2222/EdgeTPU_CPP)<br>

```
$ ./tensorflow/tensorflow/lite/tools/make/download_dependencies.sh

# For arm7vl (Raspberry Pi)
$ ./tensorflow/tensorflow/lite/tools/make/build_rpi_lib.sh

# For aarch64 (Jetson Nano or Raspberry Pi 64bit)
$ ./tensorflow/tensorflow/lite/tools/make/build_aarch64_lib.sh
```

Build module.
```
$ cd (target dir you want to build)
$ mkdir build && cd build
$ cmake ..  
$ make
```
