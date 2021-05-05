#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorFlow Lite benchmark coco metrics.

    Copyright (c) 2021 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco_detections_path",
        type=str,
        help="Path that detection result.",
        required=True,
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        help="Path that contains COCO annotations",
        required=False,
    )
    parser.add_argument(
        "--allowlist_file",
        type=str,
        help="File with COCO image ids to preprocess, one on each line.",
        default=None,
        required=False,
    )
    args = parser.parse_args()

    coco = COCO(annotation_file=args.annotation_path)
    cocoDt = coco.loadRes(args.coco_detections_path)

    # compute coco metrics
    eval = COCOeval(coco, cocoDt, "bbox")
    if args.allowlist_file is None:
        image_ids = coco.getImgIds()
    else:
        with open(args.allowlist_file, "r") as allowlist:
            image_ids = [int(x) for x in allowlist.readlines()]

    eval.params.imgIds = image_ids

    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    print(
        "AP, AP50, AP75, APsmall, APmedium, APlarge,"
        "ARmax=1, ARmax=10, ARmax=100, ARsmall, ARmidium, ARlarge"
    )
    print(eval.stats)


if __name__ == "__main__":
    main()
