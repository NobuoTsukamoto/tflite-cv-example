# Edge TPU DeepLab v3 Camouflage
Camouflage the specified label with a noise image.

## Usaege
For Raspberry Pi 3 B+ with PiCamera.
```
$ ./edge_tpu_deeplabv3_camouflage --model=<PATH_TO_MODEL_FILE> --src=pi --thread=3
```

For Jetson Nano with PiCamera.
```
$ ./edge_tpu_deeplabv3_motion --model=<PATH_TO_MODEL_FILE> --src=nano --thread=3
```

For Video stream.
```
$ ./edge_tpu_deeplabv3_motion --model=<PATH_TO_MODEL_FILE> --src=<PATH_TO_VIDEO_FILE> --thread=3
```

## Operations
- Space key: Switching the display image(segmentation image or camouflage image). 
- q key: End app.

## Full param.
```
$ ./edge_tpu_deeplabv3_camouflage --help
Usage: edge_tpu_deeplabv3_camouflage [params] 

	-?, -h, --help (value:true)
		show help command
	-H, --height (value:480)
		camera resolution height.
	-W, --width (value:640)
		camera resolution width.
	-i, --info
		display Inference fps.
	-l, --label (value:15)
		index of the target label for motion analysis. (default: Person)
	-m, --model
		path to deeplab tf-lite model flie.
	-n, --thread (value:1)
		num of thread to set tf-lite interpreter.
	-s, --src (value:nano)
		videocapture source. nano: jetson nano camera, pi: raspberry pi picamera. other: video file path
```