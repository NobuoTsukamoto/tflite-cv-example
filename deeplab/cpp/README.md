# CPP Example.

This example performs segmentation with the TensorFlow Lite C++ API using the given deeplabv3 model, and OpenCV VideoCapture IF.

Please refer to [Coral EdgeTPU C ++ API overview](https://coral.withgoogle.com/docs/edgetpu/api-cpp/) for details of C ++ API.

## How to build.
This build method targets armv7l or aarch64.<br>

Install dependency packages.<br>
**Note (Raspberry Pi): For OpenCV, install the source build with GStreamer.**
```$ sudo apt-get install build-essential
$ sudo apt install -y curl wget cmake
$ sudo apt install -y libc6-dev libc++-dev libc++abi-dev
$ sudo apt install -y libusb-1.0-0
```

Clone repository
```$ git clone https://github.com/NobuoTsukamoto/edge_tpu.git
$ cd edge_tpu/deeplab/cpp/
$ git submodule init && git submodule update
$ cd edgetpu-native/
$ git submodule init && git submodule update
```

Build TensorFlow Lite for ARM
```$ cd tensorflow
$ ./tensorflow/lite/tools/make/download_dependencies.sh
$ ./tensorflow/lite/tools/make/build_rpi_lib.sh
```

If the build is successful. Copy *libtensorflow-lite.a* file.<br>
**For arm7vl (Raspberry Pi)**
```$ ls tensorflow/lite/tools/make/gen/rpi_armv7l/lib/
benchmark-lib.a       libtensorflow-lite.a  
$ cp tensorflow/lite/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a ../../lib/libtensorflow-lite_arm32.a
```

**For aarch64 (Raspberry Pi)**
```$ ls tensorflow/lite/tools/make/gen/rpi_armv7l/lib/
benchmark-lib.a       libtensorflow-lite.a  
$ cp tensorflow/lite/tools/make/gen/aarch64_armv8-a/lib/libtensorflow-lite.a ../../lib/libtensorflow-lite_arm64.a
```

Build module.
```$ cd ../../
$ mkdir build && cd build
$ cmake ..  
$ make
```

## Usaege
For Raspberry Pi 3 B+ with PiCamera.
```
$ ./edge_tpu_deeplabv3 --model=<MODEL_FILE_PATH> --src=pi --thread=3
```

For Jetson Nano with PiCamera.
```
$ ./edge_tpu_deeplabv3 --model=<MODEL_FILE_PATH> --src=pi --thread=3
```

For Video stream.
```
$ ./edge_tpu_deeplabv3 --model=<MODEL_FILE_PATH> --src=<VIDE_FILE_PATH> --thread=3
```

Common operations
- Space key: Switching the display image(segmentation map or segmentation image). 
- q key: End app.

Full param.
```
$ ./edge_tpu_deeplabv3 --help
Usage: edge_tpu_deeplabv3 [params] 

	-?, -h, --help (value:true)
		show help command
	-H, --height (value:480)
		camera resolution height.
	-W, --width (value:640)
		camera resolution width.
	-i, --info
		display Inference fps.
	-m, --model
		path to deeplab tf-lite model flie.
	-n, --thread (value:1)
		num of thread to set tf-lite interpreter.
	-s, --src (value:nano)
		videocapture source. nano: jetson nano camera, pi: raspberry pi picamera. other: video file path
```

# Reference
- [Google Coral Edge TPU with C++ on Jetson Nano](https://qiita.com/iwatake2222/items/3a09a2d26b022a5a8a95)
- [Build TensorFlow Lite for ARM64 boards](https://www.tensorflow.org/lite/guide/build_arm64)
- [Coral EdgeTPU C++ API overview](https://coral.withgoogle.com/docs/edgetpu/api-cpp/)

