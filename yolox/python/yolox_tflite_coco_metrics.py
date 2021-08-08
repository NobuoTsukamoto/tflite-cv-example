#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorFlow Lite YOLOX with OpenCV.

    Copyright (c) 2021 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.tflite_util import get_output_tensor, make_interpreter, set_input_tensor
from yolox.utils.demo_utils import demo_postprocess, multiclass_nms

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def get_output(interpreter, score_threshold):
    """Returns list of detected objects.

    Args:
        interpreter
        score_threshold

    Returns: bounding_box, class_id, score
    """
    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    class_ids = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= score_threshold:
            result = {
                "bounding_box": boxes[i],
                "class_id": class_ids[i],
                "score": scores[i],
            }
            results.append(result)
    return results


def preprocess(image, input_size, mean, std):
    # https://github.com/Megvii-BaseDetection/YOLOX/blob/c4714bb97c2f13d26195544d5f9e1ea91241ee2b/yolox/data/data_augment.py#L165 noqa: E501

    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    image = padded_img

    image = image.astype(np.float32)
    image = image[:, :, ::-1]
    image /= 255.0
    if mean is not None:
        image -= mean
    if std is not None:
        image /= std
    # swap=(2, 0, 1)
    # image = image.transpose(swap)
    # image = np.ascontiguousarray(image, dtype=np.float32)
    return image, r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument("--thread", help="Num threads.", default=2, type=int)
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Directory containing validation set TFRecord files.",
    )
    parser.add_argument(
        "--annotation_path", type=str, help="Path that contains COCO annotations"
    )
    parser.add_argument(
        "--allowlist_file",
        type=str,
        help="File with COCO image ids to preprocess, one on each line.",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--display_every",
        type=int,
        default=100,
        help="Number of iterations executed between two consecutive display of metrics",
    )
    parser.add_argument(
        "--minival_ids", type=str, help="Path that minival ids list file.", default=None
    )
    parser.add_argument(
        "--exec_coco_metrics", help="Execute coco metrics.", action="store_true"
    )
    args = parser.parse_args()

    # Initialize TF-Lite interpreter.
    interpreter = make_interpreter(args.model, args.thread)
    interpreter.allocate_tensors()
    _, height, width, channel = interpreter.get_input_details()[0]["shape"]
    # _, channel, height, width = interpreter.get_input_details()[0]["shape"]
    input_shape = (height, width)
    print("Interpreter(height, width, channel): ", height, width, channel)

    # COCO Datasets.
    coco = COCO(annotation_file=args.annotation_path)
    class_ids = sorted(coco.getCatIds())

    # Image IDs
    if args.allowlist_file is None:
        image_ids = coco.getImgIds()
    else:
        with open(args.allowlist_file, "r") as allowlist:
            image_ids = set([int(x) for x in allowlist.readlines()])

    num_steps = len(image_ids)
    elapsed_list = []
    coco_detections = []

    for i, image_id in enumerate(image_ids):
        coco_img = coco.imgs[image_id]

        # Load image.
        im = cv2.imread(os.path.join(args.images_dir, coco_img["file_name"]))
        im, ratio = preprocess(im, input_shape, mean, std)

        # Run inference.
        start = time.perf_counter()

        set_input_tensor(interpreter, im)
        interpreter.invoke()
        output_details = interpreter.get_output_details()[0]
        output = interpreter.get_tensor(output_details["index"])

        inference_time = time.perf_counter() - start

        predictions = demo_postprocess(output, input_shape, p6=args.with_p6)[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.65, score_thr=0.01)
        elapsed_list.append(inference_time)

        # Display result.
        if dets is not None:
            final_boxes = dets[:, :4]
            final_scores = dets[:, 4]
            final_cls_inds = dets[:, 5]

            for j, box in enumerate(final_boxes):
                class_id = int(final_cls_inds[j])
                score = final_scores[j]

                xmin = int(box[0])
                ymin = int(box[1])
                xmax = int(box[2])
                ymax = int(box[3])

                bbox_coco_fmt = [
                    xmin,  # x0
                    ymin,  # x1
                    (xmax - xmin),  # width
                    (ymax - ymin),  # height
                ]
                coco_detection = {
                    "image_id": image_id,
                    "category_id": class_ids[class_id],
                    "bbox": [int(coord) for coord in bbox_coco_fmt],
                    "score": float(score),
                }
                coco_detections.append(coco_detection)

        if (i + 1) % args.display_every == 0:
            print(
                "  step %03d/%03d, iter_time(ms)=%.0f"
                % (i + 1, num_steps, elapsed_list[-1] * 1000)
            )

    # write coco detections to file
    coco_detections_path = os.path.join(".", "coco_detections.json")
    with open(coco_detections_path, "w") as f:
        json.dump(coco_detections, f)

    cocoDt = coco.loadRes(coco_detections_path)

    if args.exec_coco_metrics is not None:
        # compute coco metrics
        eval = COCOeval(coco, cocoDt, "bbox")
        eval.params.imgIds = image_ids

        eval.evaluate()
        eval.accumulate()
        eval.summarize()

        os.remove(coco_detections_path)


if __name__ == "__main__":
    main()
