"""Test the conversions module."""

import numpy as np
from com_iseg import conversions


def test_inversion():
    size = 10
    lbl = np.zeros((size, size), dtype=np.uint8)
    mid = size // 2
    sl = slice(mid, -mid)
    lbl[sl, sl] = 1

    descriptors = conversions.lbl_to_local_descriptors(lbl)
    new_lbl = conversions.local_descriptors_to_lbl(descriptors)
    new_descriptors = conversions.lbl_to_local_descriptors(new_lbl)

    np.testing.assert_array_equal(descriptors, new_descriptors)
