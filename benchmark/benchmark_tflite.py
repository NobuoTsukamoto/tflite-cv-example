#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorFlow Lite benchmark.

    Copyright (c) 2021 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import os
import argparse
import time

import numpy as np
from utils.tflite_util import get_output_tensor, make_interpreter, set_input_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument("--thread", help="Num threads.", default=2, type=int)
    parser.add_argument("--count", help="Repeat count.", default=1000, type=int)
    parser.add_argument(
        "--display_every",
        type=int,
        default=100,
        help="Number of iterations executed between two consecutive display of metrics",
    )
    parser.add_argument(
        "--num_warmup_iterations",
        type=int,
        default=50,
        help="Number of initial iterations skipped from timing",
    )
    parser.add_argument(
        "--target_duration",
        type=int,
        default=None,
        help="If set, script will run for specified number of seconds.",
    )
    args = parser.parse_args()

    print("Model name: {0}, param num_threads: {1}".format(
        os.path.splitext(os.path.basename(args.model))[0],
        args.thread))

    # Initialize TF-Lite interpreter.
    interpreter = make_interpreter(args.model, args.thread)
    interpreter.allocate_tensors()
    batch, height, width, channel = interpreter.get_input_details()[0]["shape"]
    print("Interpreter: ", batch, height, width, channel)

    elapsed_list = []

    for i in range(args.count):
        input_im = np.random.randint(0, 256, (height, width, channel))

        # Run inference.
        start = time.perf_counter()

        set_input_tensor(interpreter, input_im)
        interpreter.invoke()

        inference_time = time.perf_counter() - start
        elapsed_list.append(inference_time)

        if i == 0:
            print("First Inference : {0:.2f} ms".format(inference_time * 1000))

        if (i + 1) % args.display_every == 0:
            print(
                "  step %03d/%03d, iter_time(ms)=%.0f"
                % (i + 1, args.count, elapsed_list[-1] * 1000)
            )      

    results = {}
    iter_times = np.array(elapsed_list)
    results["total_time"] = np.sum(iter_times)
    iter_times = iter_times[args.num_warmup_iterations :]
    results["images_per_sec"] = np.mean(batch / iter_times)
    results["99th_percentile"] = (
        np.percentile(iter_times, q=99, interpolation="lower") * 1000
    )
    results["latency_mean"] = np.mean(iter_times) * 1000
    results["latency_median"] = np.median(iter_times) * 1000
    results["latency_min"] = np.min(iter_times) * 1000

    print("  images/sec: %d" % results["images_per_sec"])
    print("  99th_percentile(ms): %.2f" % results["99th_percentile"])
    print("  total_time(s): %.1f" % results["total_time"])
    print("  latency_mean(ms): %.2f" % results["latency_mean"])
    print("  latency_median(ms): %.2f" % results["latency_median"])
    print("  latency_min(ms): %.2f" % results["latency_min"])


if __name__ == "__main__":
    main()
