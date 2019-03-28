#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Copyright (c) 2018 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.

    Downloads and converts Oxford 17 Category Flower dataset to TF-Example protos.

    This modult downloads the Oxford 17 Category Flower Dataset, uncompresses it,
    reads the files that make up the Flowers data and creates two TFRecord dataset:
    one for train and one for test. Each TFRecord dataset is comprised of a set of
    TF-Example protocol buffers, each of which contain a single image and label.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import argparse
import tarfile
import glob
import shutil

from PIL import Image, ImageOps

from six.moves import urllib
import tensorflow as tf

# The URL where the Flowers data can be downloaded.
_DATA_URL = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz'

# The number of images in the validation set.
_NUM_VALIDATION = 136

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5

# The Class labels
CLASS_NAMES = ['Tulip', 'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus', 'Iris',\
            'Tigerlily', 'Daffodil', 'Fritillary', 'Sunflower', 'Daisy', 'ColtsFoot',\
            'Dandelion', 'Cowslip', 'Buttercup', 'Widnflower', 'Pansy']

class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilites.
    """
    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_filenames_and_classes(dataset_dir, is_data_augmentation=False):
    """ Returns a dictionary of file names and inferred class names.

    Args:
        dataset_dir: A directory containing a JPG encoded images.
        is_data_augmantation: If True, Image data augmantation. flip image horizontal
    Returns:
        A dictionary of class names and file paths and representing class names.
    """
    flower_root = os.path.join(dataset_dir, 'jpg')

    files = glob.glob(os.path.join(flower_root, '*.jpg'))
    files.sort()

    par_category = 80
    photo_filename = []
    for i, class_name in enumerate(CLASS_NAMES):
        direcotry = os.path.join(dataset_dir, class_name)
        os.mkdir(direcotry)
        for j in range(par_category):
            index = (par_category * i) + j
            shutil.move(files[index], direcotry)
            file_path = os.path.join(direcotry, os.path.basename(files[index]))
            photo_filename.append(file_path)

            # Data augmantation.
            if is_data_augmentation:
                im = Image.open(file_path)
                mirror_im = ImageOps.mirror(im)
                flip_path = os.path.join(direcotry,
                        os.path.splitext(os.path.basename(files[index]))[0] + '_flip.jpg')
                mirror_im.save(flip_path)
                photo_filename.append(flip_path)


    return photo_filename, sorted(CLASS_NAMES)

def _get_dataset_filename(dataset_dir, split_name, shard_id):
    """ Return TF-Recode file name.

    Args:
        dataset_dir: dataset direcotry.
        split_name: train or validation.
        shard_id: id.
    Returns:
        TF-Recode file path.
    """
    output_filename = 'flowers_%s_%05d-of-%05d.tfrecord' % (
                      split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)

def int64_feature(values):
    """Returns a TF-Feature of int64s.
    Args:
        values: A scalar or list of values.
    Returns:
        A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.
    Args:
        values: A string.
    Returns:
        A TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def _image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))

def _conver_dataset(split_name, file_names, class_names_to_ids, dataset_dir):
    """Converts the given file names to a TF-Recode dataset.

    Args:
        split_name: The name of thedataset, either 'train' or 'validation'.
        file_names: A List of absolute paths to jpg images.
        class_names_to_ids: A dictionary from class names (strings) to ids (integers).
        dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(file_names) / float(_NUM_SHARDS)))
    print(num_per_shard)

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(_NUM_SHARDS):
                output_file_name = _get_dataset_filename(dataset_dir,
                        split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_file_name) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(file_names))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' %
                                         (i + 1, len(file_names), shard_id))
                        sys.stdout.flush()

                        # Read the file name.
                        image_data = tf.gfile.GFile(file_names[i], 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)

                        class_name = os.path.basename(os.path.dirname(file_names[i]))
                        class_id = class_names_to_ids[class_name]

                        example = _image_to_tfexample(image_data, b'jpg',
                                height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()

def _clean_up_temporary_files(dataset_dir):
    """Removes temporary files used to create the dataset.

    Args:
        dataset_dir: The directory where the temporary files are stoerd.
    """
    file_name = _DATA_URL.split('/')[-1]
    file_path = os.path.join(dataset_dir, file_name)
    tf.gfile.Remove(file_path)

    tmp_dir = os.path.join(dataset_dir, 'jpg')
    tf.gfile.DeleteRecursively(tmp_dir)

    for sub_dir in CLASS_NAMES:
        tf.gfile.DeleteRecursively(os.path.join(dataset_dir, sub_dir))

def _dataset_exists(dataset_dir):
    """ Check if TF-Recoed exists.

    Args:
        dataset_dir: dataset directory.
    Returns:
        True: TF-Recode are exist.
        False: TF-Recodes are not exist.
    """
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS):
            output_file_name = _get_dataset_filename(dataset_dir, split_name, shard_id)
            if not tf.gfile.Exists(output_file_name):
                return False
    return True

def _download_and_uncompress_tarball(tarball_url, dataset_dir):
    """ Downloads the 'tarball_url' and uncompresses it locally.

    Args:
        tarball_url: The URL of a tarball file.
        dataset_dir: The directory where the temporary files are stored.
    """
    file_name = tarball_url.split('/')[-1]
    file_path = os.path.join(dataset_dir, file_name)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r >> Downloading %s %.1f%%' % (file_name,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    file_path, _ = urllib.request.urlretrieve(tarball_url, file_path, _progress)
    print()
    statinfo = os.stat(file_path)
    print('Successfully downladed', file_name, statinfo.st_size, 'bytes.')
    tarfile.open(file_path, 'r:gz').extractall(dataset_dir)

def _write_label_file(labels_to_class_names, dataset_dir, filename='labels.txt'):
    """Writes a file with the list of class names.

    Args:
        labels_to_class_names: A map of (integer) labels to class names.
        dataset_dir: The directory in which the labels file should be written.
        filename: The filename where the class names are written.
    """
    labels_file_name = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_file_name, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', help='Path to dataset dir.')
    parser.add_argument('--flip', action='store_true')
    args = parser.parse_args()

    if not tf.gfile.Exists(args.dataset_dir):
        tf.gfile.MakeDirs(args.dataset_dir)

    if _dataset_exists(args.dataset_dir):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    _download_and_uncompress_tarball(_DATA_URL, args.dataset_dir)
    photo_file_names, CLASS_NAMES = _get_filenames_and_classes(args.dataset_dir, args.flip)
    class_names_to_ids = dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))

    # Divide into train and test.
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_file_names)
    training_file_names = photo_file_names[_NUM_VALIDATION:]
    validation_file_names = photo_file_names[:_NUM_VALIDATION]

    # First, convert the training and validation sets.
    _conver_dataset('train', training_file_names, class_names_to_ids,
            args.dataset_dir)
    _conver_dataset('validation', validation_file_names, class_names_to_ids,
            args.dataset_dir)

    # Finally, write the labels file.
    labels_to_class_names = dict(zip(range(len(CLASS_NAMES)), CLASS_NAMES))
    _write_label_file(labels_to_class_names, args.dataset_dir)

    _clean_up_temporary_files(args.dataset_dir)
    print('\nFinished converting the 17 Flowers dataset!')

if __name__ == '__main__':
    main()

