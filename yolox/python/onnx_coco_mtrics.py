#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import json

import cv2
import numpy as np

import onnxruntime

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# from yolox.data.data_augment import preproc as preprocess
from yolox.utils.demo_utils import multiclass_nms, demo_postprocess


def preprocess(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR
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
    image = image.transpose(swap)
    image = np.ascontiguousarray(image, dtype=np.float32)
    return image, r


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
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
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()

    # COCO Datasets.
    coco = COCO(annotation_file=args.annotation_path)
    class_ids = sorted(coco.getCatIds())
    image_ids = coco.getImgIds()

    input_shape = tuple(map(int, args.input_shape.split(',')))
    session = onnxruntime.InferenceSession(args.model)

    num_steps = len(image_ids)
    coco_detections = []
    elapsed_list = []
    coco_detections = []

    for i, image_id in enumerate(image_ids):
        coco_img = coco.imgs[image_id]

        origin_img = cv2.imread(os.path.join(args.images_dir, coco_img["file_name"]))
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        img, ratio = preprocess(origin_img, input_shape, mean, std)

        ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
        output = session.run(None, ort_inputs)
        predictions = demo_postprocess(output[0], input_shape, p6=args.with_p6)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.65, score_thr=0.01)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]

            for j in range(len(final_boxes)):
                box = final_boxes[j]
                cls_id = int(final_cls_inds[j])
                score = final_scores[j]

                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])

                bbox_coco_fmt = [
                    x0,  # x0
                    y0,  # x1
                    (x1 - x0),  # width
                    (y1 - y0),  # height
                ]
                coco_detection = {
                    "image_id": image_id,
                    "category_id": class_ids[cls_id],
                    "bbox": [int(coord) for coord in bbox_coco_fmt],
                    "score": float(score),
                }
                coco_detections.append(coco_detection)

        if (i + 1) % 1000 == 0:
            print(
                "  step %03d/%03d"
                % (i + 1, num_steps)
            )

    # write coco detections to file
    coco_detections_path = os.path.join(".", "coco_detections.json")
    with open(coco_detections_path, "w") as f:
        json.dump(coco_detections, f)

    cocoDt = coco.loadRes(coco_detections_path)

    # compute coco metrics
    eval = COCOeval(coco, cocoDt, "bbox")
    eval.params.imgIds = image_ids

    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    os.remove(coco_detections_path)
