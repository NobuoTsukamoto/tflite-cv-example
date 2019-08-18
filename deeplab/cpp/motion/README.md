# Edge TPU DeepLab v3 Afterimage.
Draws the afterimage of the specified label. 

## Usaege
For Raspberry Pi 3 B+ with PiCamera.
```
$ ./edge_tpu_deeplabv3_motion --model=<PATH_TO_MODEL_FILE> --src=pi --thread=3
```

For Jetson Nano with PiCamera.
```
$ ./edge_tpu_deeplabv3_motion --model=<PATH_TO_MODEL_FILE> --src=nano --thread=3
```

For Video stream.
```
$ ./edge_tpu_deeplabv3_motion --model=<PATH_TO_MODEL_FILE> --src=<PATH_TO_VIDEO_FILE> --thread=3
```

ex.<br>
(label=5:bottol, skipped frame=10, afterimage count=5, thread=3)
```
./edge_tpu_deeplabv3_motion --model=../../model/deeplabv3_mnv2_dm05_pascal_trainaug_edgetpu.tflite --src=nano --thread=3 --label=5 --skip=10 --count=5
```

## Operations
- q key: End app.

## Full param.
```
$ ./edge_tpu_deeplabv3_motion --help
Usage: edge_tpu_deeplabv3_motion [params] 

	-?, -h, --help (value:true)
		show help command.
	-H, --height (value:480)
		camera resolution height.
	-S, --skip (value:0)
		number of skipped frames.
	-W, --width (value:640)
		camera resolution width.
	-c, --count (value:10)
		number of mask image histories.
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