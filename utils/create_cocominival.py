#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorFlow Lite benchmark.

    Copyright (c) 2021 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""


import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--instances_file',
      type=str,
      help='Full path of the input JSON file, like instances_val20xx.json.',
      required=True)
    parser.add_argument(
      '--allowlist_file',
      type=str,
      help='File with COCO image ids to preprocess, one on each line.',
      required=False)
    args = parser.parse_args()

    # Read JSON data into a dict.
    with open(args.instances_file) as f:
        df = json.load(f)

    with open(args.allowlist_file, 'r') as allowlist:
        image_id_allowlist = set([int(x) for x in allowlist.readlines()])

    print(len(df['images']))
    for image_dict in reversed(range(len(df['images']))):
        image_id = df['images'][image_dict]['id']
        if image_id not in image_id_allowlist:
            df['images'].pop(image_dict)
    print(len(df['images']))

    # print(len(df['annotations']))
    # for image_dict in reversed(range(len(df['annotations']))):
    #     image_id = df['annotations'][image_dict]['image_id']
    #     if image_id not in image_id_allowlist:
    #         df['annotations'].pop(image_dict)
    # print(len(df['annotations']))

    with open('./result.json', 'w') as f:
        json.dump(df, f,)


if __name__ == "__main__":
    main()
