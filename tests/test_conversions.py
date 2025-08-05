"""Test the conversions module."""

import numpy as np
from com_iseg import conversions


def test_negative_lbls():
    size = 12
    lbl = np.zeros((size, size), dtype=np.int8)
    start = size // 4
    sl = slice(start, start + size // 2)
    lbl[sl, sl] = -1

    descriptors = conversions.lbl_to_local_descriptors(lbl)
    assert np.all(descriptors == 0.0)


def test_probability_of_local_descriptors():
    size = 12
    lbl = np.zeros((size, size), dtype=np.uint8)
    start = size // 4
    sl = slice(start, start + size // 2)
    lbl[sl, sl] = 2  # skip lbl = 1 to trigger 'if obj is None'

    descriptors = conversions.lbl_to_local_descriptors(lbl)
    np.testing.assert_array_equal(lbl == 2, descriptors[0] == 1.0)


def test_local_descriptors():
    size = 12
    lbl = np.zeros((size, size), dtype=np.uint8)
    start = size // 4
    sl = slice(start, start + size // 2)
    lbl[sl, sl] = 2  # skip lbl = 1 to trigger 'if obj is None'

    descriptors = conversions.lbl_to_local_descriptors(lbl)

    inds_foreground = np.any(descriptors[1:] != 0.0, axis=0)
    assert np.all(descriptors[0, inds_foreground] == 1.0)

    inds_background = descriptors[0] == 0.0
    assert np.all(descriptors[1:, inds_background] == 0.0)


def test_inversion():
    size = 12
    lbl = np.zeros((size, size), dtype=np.uint8)
    start = size // 4
    sl = slice(start, start + size // 2)
    lbl[sl, sl] = 2  # skip lbl = 1 to trigger 'if obj is None'

    descriptors = conversions.lbl_to_local_descriptors(lbl)
    new_lbl = conversions.local_descriptors_to_lbl(descriptors)
    new_descriptors = conversions.lbl_to_local_descriptors(new_lbl)

    np.testing.assert_array_equal(descriptors, new_descriptors)


def test_mask():
    size = 10
    lbl = np.zeros((size, size), dtype=np.uint8)

    lbl[:4, :4] = 1
    lbl[1:4, 7:8] = 2
    lbl[8:, 8:] = 4  # skip lbl = 1 to trigger 'if obj is None'

    mask = conversions.mask_bordering_lbls(lbl)

    np.testing.assert_array_equal(np.logical_or(lbl == 0, lbl == 2), mask)
