# Edge TPU DeepLab v3 Benchmark
Benchmark to measure inference time.  

## Usaege
```
$ ./edge_tpu_deeplabv3 <PATH_TO_IMAGE_FILE> --model=<PATH_TO_MODEL_FILE>
```

## Full param.
```
$ ./edge_tpu_deeplabv3_benchmark --help
Usage: edge_tpu_deeplabv3_benchmark [params] input 

	-?, -h, --help (value:true)
		show help command
	-d, --detail
		output log for each inference.
	-i, --iterations (value:20)
		number of inference iterations.
	-m, --model
		path to deeplab tf-lite model flie.
	-n, --thread (value:1)
		num of thread to set tf-lite interpreter.

	input
		path to input image file.
```