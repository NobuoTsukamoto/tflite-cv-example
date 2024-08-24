# FFNet TensorFlow Lite GPU Delegate VisionFive 2.

## HW
- [VisionFive 2](https://www.starfivetech.com/en/site/boards) 202306 Release

## How to build.

### Clone repository and init submodule.
```
$ git clone https://github.com/NobuoTsukamoto/tflite-cv-example.git
$ cd tflite-cv-example
$ git submodule update --init --recursive
```

### Build 
```
$ cd ffnet
$ mkdir build && cd build
$ cmake ..
$ make -j3
```

## Download FFNet model from PINTO_model_zoo
```
$ git clone https://github.com/PINTO0309/PINTO_model_zoo.git
$ cd 395_FFNet
$ ./download.sh
```

## Usage
```
$ ./ffnet_tflite_gpu \
  --model=_PATH_TO_MODEL_FILE_ \
  --video=_PATH_TO_INPUT_VIDEO_FILE_ \
  --output=_PATH_TO_OUTPUT_VIDEO_FILE_
```

## Refferince
- [FFNet: Simple and Efficient Architectures for Semantic Segmentation](https://github.com/Qualcomm-AI-research/FFNet)
- [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
- [onnx2tf](https://github.com/PINTO0309/onnx2tf)
