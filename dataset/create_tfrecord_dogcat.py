#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Copyright (c) 2018 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.

    Converts Dog vs Cat dataset to TF-Example protos.

    This module uncompresses the dog vs cat dataset,
    reads the files that make up the data and creates two TFRecord dataset:
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
import zipfile

from PIL import Image, ImageOps

from six.moves import urllib
import tensorflow as tf

# The nunber of images in the train set.
_NUM_TRAIN = 2000

# The number of images in the validation set.
_NUM_VALIDATION = 500

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5

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


def _get_filenames_and_classes(dataset_dir, num_train=2000):
    """ Returns a dictionary of file names and inferred class names.

    Args:
        dataset_dir: A directory containing a JPG encoded images.
        num_train: The nunber of images in the train set.
    Returns:
        A dictionary of class names and file paths and representing class names.
    """
    pat_root = os.path.join(dataset_dir, 'train')
    directories = []
    class_names = []
    for filename in os.listdir(pat_root):
        path = os.path.join(pat_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames = []
    for directory in directories:
        for i, filename in enumerate(os.listdir(directory)):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)
            if i > num_train:
                break

    return photo_filenames, sorted(class_names)

def _get_dataset_filename(dataset_dir, split_name, shard_id):
    """ Return TF-Recode file name.

    Args:
        dataset_dir: dataset directory.
        split_name: train or validation.
        shard_id: id.
    Returns:
        TF-Recode file path.
    """
    output_filename = 'dogcat_%s_%05d-of-%05d.tfrecord' % (
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
    file_path = os.path.join(dataset_dir, 'test.zip')
    tf.gfile.Remove(file_path)

    file_path = os.path.join(dataset_dir, 'train.zip')
    tf.gfile.Remove(file_path)

    file_path = os.path.join(dataset_dir, 'sample_submission.csv')
    tf.gfile.Remove(file_path)

    tmp_dir = os.path.join(dataset_dir, 'train')
    tf.gfile.DeleteRecursively(tmp_dir)

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

def _uncompress_zip(zip_file, dataset_dir):
    """ Uncompresses zip file locally.

    Args:
        zip_file: A zip file pth.
        dataset_dir: The directory where the temporary files are stored.
    """
    print('Unzip : ', zip_file)
    with zipfile.ZipFile(zip_file) as existing_zip:
        existing_zip.extractall(dataset_dir)

    print('Unzip : ', os.path.join(dataset_dir, 'train.zip'))
    with zipfile.ZipFile(os.path.join(dataset_dir, 'train.zip')) as existing_zip:
        existing_zip.extractall(dataset_dir)

    print('Create directory and move files.')
    train_path = os.path.join(dataset_dir, 'train')

    # move cat.
    os.mkdir(os.path.join(train_path, 'cat'))
    files = glob.glob(os.path.join(train_path, 'cat.*'))
    for path in files:
        file_name = os.path.basename(path)
        shutil.move(os.path.join(train_path, file_name), os.path.join(train_path, 'cat/'))

    # move dog.
    os.mkdir(os.path.join(train_path, 'dog'))
    files = glob.glob(os.path.join(train_path, 'dog.*'))
    for path in files:
        file_name = os.path.basename(path)
        shutil.move(os.path.join(train_path, file_name), os.path.join(train_path, 'dog/'))

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
    parser.add_argument('zip_file', help='Path to data zip file.')
    parser.add_argument('dataset_dir', help='Path to dataset dir.')
    args = parser.parse_args()

    if not tf.gfile.Exists(args.dataset_dir):
        tf.gfile.MakeDirs(args.dataset_dir)

    if _dataset_exists(args.dataset_dir):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    _uncompress_zip(args.zip_file, args.dataset_dir)
    photo_file_names, CLASS_NAMES = _get_filenames_and_classes(args.dataset_dir)
    class_names_to_ids = dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))

    # Divide into train and test.
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_file_names)

    num_split = _NUM_VALIDATION
    training_file_names = photo_file_names[num_split:]
    validation_file_names = photo_file_names[:num_split]

    # First, convert the training and validation sets.
    _conver_dataset('train', training_file_names, class_names_to_ids,
            args.dataset_dir)
    _conver_dataset('validation', validation_file_names, class_names_to_ids,
            args.dataset_dir)

    # Finally, write the labels file.
    labels_to_class_names = dict(zip(range(len(CLASS_NAMES)), CLASS_NAMES))
    _write_label_file(labels_to_class_names, args.dataset_dir)

    _clean_up_temporary_files(args.dataset_dir)
    print('\nFinished converting the Dog vs Cat dataset!')

if __name__ == '__main__':
    main()

