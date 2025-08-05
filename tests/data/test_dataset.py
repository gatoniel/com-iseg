"""Test COMDataset and its utility functions."""

import numpy as np
from com_iseg.data import dataset


def test_get_masks():
    size = 20
    lbl = np.zeros((2, size, size), dtype=np.int8)

    lbl[0, :4, :4] = 1
    lbl[0, 1:4, 7:8] = 2
    lbl[0, 11:14, 17:18] = -3
    lbl[0, 8:, 8:] = 4  # skip lbl = 1 to trigger 'if obj is None'

    lbl[1, 4:8, 10:15] = -1

    mask = dataset.get_masks(lbl)

    np.testing.assert_array_equal(np.logical_or(lbl == 0, lbl == 2), mask)
