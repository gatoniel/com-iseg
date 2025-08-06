"""Test COMDataset and its utility functions."""

import numpy as np
from com_iseg.data import dataset
from com_iseg import conversions


def test_get_masks():
    size = 20
    lbl = np.zeros((2, size, size), dtype=np.int8)

    lbl[0, :4, :4] = 1
    lbl[0, 1:4, 7:8] = 2
    lbl[0, 11:14, 13:15] = -3
    lbl[0, 17:, 17:] = 4  # skip lbl = 1 to trigger 'if obj is None'

    assert np.any(lbl[0] < 0)

    lbl[1, 4:8, 10:15] = -1

    mask = dataset.get_masks(lbl)

    np.testing.assert_array_equal(np.logical_or(lbl == 0, lbl == 2), mask)


def test_len_COMDataset():
    size = 30
    lbl = np.zeros((2, size, size, size), dtype=np.int8)
    img = np.empty((2, size, size, size), dtype=float)

    lbl[0, :4, :4, 10:14] = 1
    lbl[0, 1:4, 7:8, 10:14] = 2
    lbl[0, 11:14, 17:18, 10:14] = -3
    lbl[0, 8:, 8:, 10:14] = 4  # skip lbl = 1 to trigger 'if obj is None'

    lbl[1, 4:8, 10:15, 20:25] = -1

    imgs = [img[i] for i in range(2)]
    lbls = [lbl[i] for i in range(2)]

    ds = dataset.COMDataset(imgs, lbls, (16, 16, 14))

    #  two images, two patches in the first two dimensions, three patches in last
    #  dimension
    assert len(ds) == 2 * 2 * 2 * 3


def test_patch_size_COMDataset():
    size = 30
    lbl = np.zeros((2, size, size, size), dtype=np.int8)

    lbl[0, :4, :4, 10:14] = 1
    lbl[0, 1:4, 7:8, 10:14] = 2
    lbl[0, 11:14, 17:18, 10:14] = -3
    lbl[0, 8:, 8:, 10:14] = 4  # skip lbl = 1 to trigger 'if obj is None'

    lbl[1, 4:8, 10:15, 20:25] = -1

    lbls = [lbl[i] for i in range(2)]

    patch_size = (16, 16, 14)
    ds = dataset.COMDataset([lbl.astype(float) for lbl in lbls], lbls, patch_size)

    for patches in ds:
        assert patches[0].shape == (1,) + patch_size
        assert patches[1].shape == (1,) + patch_size
        assert patches[2].shape == (3,) + patch_size
        assert patches[3].shape == patch_size


def test_exactly_one_patch():
    size = 32
    lbl = np.zeros((size, size, size), dtype=np.int8)

    patch_size = (size, size, size)
    ds = dataset.COMDataset([lbl.astype(float)], [lbl], patch_size)

    for patches in ds:
        assert patches[0].shape == (1,) + patch_size
        assert patches[1].shape == (1,) + patch_size
        assert patches[2].shape == (3,) + patch_size
        assert patches[3].shape == patch_size


def test_patch_dtype_COMDataset():
    size = 30
    lbl = np.zeros((2, size, size, size), dtype=np.int8)

    lbl[0, :4, :4, 10:14] = 1
    lbl[0, 1:4, 7:8, 10:14] = 2
    lbl[0, 11:14, 17:18, 10:14] = -3
    lbl[0, 8:, 8:, 10:14] = 4  # skip lbl = 1 to trigger 'if obj is None'

    lbl[1, 4:8, 10:15, 20:25] = -1

    lbls = [lbl[i] for i in range(2)]

    patch_size = (16, 16, 14)
    ds = dataset.COMDataset([lbl.astype(float) for lbl in lbls], lbls, patch_size)

    for patches in ds:
        for patch in patches[:3]:
            assert patch.dtype == np.float32
        assert patches[3].dtype == bool


def test_tiling_COMDataset():
    size = 30
    lbl = np.zeros((2, size, size, size), dtype=np.int8)

    lbl[0, :4, :4, 10:14] = 1
    lbl[0, 1:4, 7:8, 10:14] = 2
    lbl[0, 11:14, 17:18, 10:14] = -3
    lbl[0, 21:, 21:, 10:14] = 4  # skip lbl = 1 to trigger 'if obj is None'

    lbl[1, 4:8, 10:15, 20:25] = -1

    lbls = [lbl[i] for i in range(2)]

    patch_size = (16, 16, 14)
    ds = dataset.COMDataset(
        [lbl.astype(float) for lbl in lbls],
        lbls,
        patch_size,
        normalize=False,
        clip_eps=None,
    )

    for patches in ds:
        # mask is correct
        tmp_lbl = patches[0].astype(int)
        mask = dataset.get_masks(tmp_lbl)
        np.testing.assert_array_equal(mask[0], patches[3])

        # lbl and com descriptors are the same?
        new_lbl = conversions.local_descriptors_to_lbl(
            np.concatenate([patches[1], patches[2]], axis=0)
        )
        lbl_new = {}
        for i, j in zip(tmp_lbl.flatten(), new_lbl.flatten()):
            try:
                lbl_new[(i, j)] += 1
            except KeyError:
                lbl_new[(i, j)] = 1

        for i in np.unique(tmp_lbl):
            assert len([None for (j, _) in lbl_new if j == i]) == 1

        for i in np.unique(new_lbl):
            if i != 0:
                assert len([None for (_, j) in lbl_new if j == i]) == 1
        assert len([None for (k, j) in lbl_new if j == 0 and k > 0]) == 0
