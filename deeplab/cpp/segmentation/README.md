# Edge TPU DeepLab v3 Segmentation
Draw the detected label with a color map.

## Usaege
For Raspberry Pi 3 B+ with PiCamera.
```
$ ./edge_tpu_deeplabv3 --model=<PATH_TO_MODEL_FILE> --src=pi --thread=3
```

For Jetson Nano with PiCamera.
```
$ ./edge_tpu_deeplabv3 --model=<PATH_TO_MODEL_FILE> --src=pi --thread=3
```

For Video stream.
```
$ ./edge_tpu_deeplabv3 --model=<PATH_TO_MODEL_FILE> --src=<PATH_TO_VIDEO_FILE> --thread=3
```

## Operations
- Space key: Switching the display image(segmentation map or segmentation image). 
- q key: End app.

## Full param.
```
$ ./edge_tpu_deeplabv3 --help
Usage: edge_tpu_deeplabv3 [params] 

	-?, -h, --help (value:true)
		show help command.
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