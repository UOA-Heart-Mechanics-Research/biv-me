#!/usr/bin/env python

import itertools

import numpy as np


PADDING = 1


# Stolen from https://stackoverflow.com/a/31402351/5902284
def compute_bounding_box(img):
    n = img.ndim
    out = []
    for ax in itertools.combinations(range(n), n - 1):
        nonzero = np.any(img, axis=ax)
        out.append(slice(*(np.where(nonzero)[0][[0, -1]] + [0, 1])))
    return tuple(out)[::-1]


def crop_array(nda_image, bbox_from=None, pad_values=0):
    if bbox_from is None:
        bbox_from = nda_image
    bounding_box = compute_bounding_box(bbox_from)
    cropped_image = nda_image[bounding_box]
    padded_image = np.pad(cropped_image, PADDING, 'constant',
                          constant_values=pad_values)
    restore_padding = []
    for slice_, shape in zip(bounding_box, nda_image.shape):
        restore_padding += [(slice_.start, shape - slice_.stop)]
    return padded_image, tuple(restore_padding)


def restore_array(image, restore_padding):
    return np.pad(image[(slice(PADDING, -PADDING),) * image.ndim],
                  pad_width=restore_padding,
                  mode="constant")
