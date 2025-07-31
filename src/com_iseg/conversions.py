"""The functions to convert between lbl map and descriptor fields."""

import numpy as np
from scipy import ndimage
from numpy import typing as npt
from scipy.spatial import KDTree
from tqdm import tqdm


def calc_moments_binary(
    lbl: npt.NDArray[np.bool],
) -> tuple[
    npt.NDArray[np.double],
    npt.NDArray[np.int_],
]:
    """Calculate center point and coordinates for boolean mask."""
    coordinates = np.argwhere(lbl)
    center_point = np.mean(coordinates, axis=0)
    return center_point, coordinates


def lbl_to_local_descriptors(
    lbl: npt.NDArray[np.int_],
) -> npt.NDArray[np.double]:
    """Transfer label information into local descriptors, e.g., midpoints."""
    descriptors = np.zeros(lbl.shape + (1 + lbl.ndim,), dtype=float)

    objects = ndimage.find_objects(lbl)
    for i, obj in enumerate(objects):
        if obj is None:
            continue

        lbl_id = i + 1

        offset = np.array([obj[i].start for i in range(lbl.ndim)])

        mask = lbl[obj] == lbl_id
        center_point, coordinates = calc_moments_binary(mask)
        global_coordinates = coordinates + offset
        inds = tuple(global_coordinates[:, i] for i in range(lbl.ndim))

        descriptors[inds + (slice(1, None),)] = center_point - coordinates
        descriptors[inds + (0,)] = 1.0

    return descriptors


def local_descriptors_to_lbl(descriptors, max_dist=1):
    """Simple function to calculate label objects from description by offsets."""
    lbl = np.zeros((descriptors.shape[:-1]), dtype=np.uint16)

    global_coordinates = np.argwhere(descriptors[..., 0] > 0)

    inds = tuple(global_coordinates[:, i] for i in range(lbl.ndim))

    coords = descriptors[inds + (slice(1, None),)] + global_coordinates
    tree = KDTree(coords)

    lbl_id = 1
    iterator = LinearSearchIterator(coords.shape[0])
    for i in tqdm(iterator):
        indices = tree.query_ball_point(coords[i], max_dist)

        iterator.set_false(indices)

        inds = tuple(global_coordinates[indices, j] for j in range(lbl.ndim))

        lbl[inds] = lbl_id
        lbl_id += 1

    return lbl


class LinearSearchIterator:
    """A search iterator."""

    def __init__(self, size):
        """Set indexer based on given size and initialize first index with zero."""
        self.size = size
        self.indexer = np.ones(size, dtype=bool)
        self.i = 0

    def set_false(self, indices):
        """Set indices to False so they will get skipped at the next search."""
        self.indexer[indices] = False

    def __iter__(self):
        """Return for usage as iterator."""
        return self

    def __next__(self):
        """Find the next index where self.indexer is True."""
        self.i = search_next_index(self.i, self.indexer)
        return self.i

    def __len__(self):
        return self.size


def search_next_index(current_index, indexer):
    """Find next index of indexer array that is True."""
    while current_index < len(indexer):
        if indexer[current_index]:
            return current_index
        current_index += 1

    raise StopIteration
