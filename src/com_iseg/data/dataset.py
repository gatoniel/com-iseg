"""Dataset class to organize training preparation."""

from itertools import product
import numpy as np
from torch.utils.data import Dataset
from ..conversions import lbl_to_local_descriptors, mask_bordering_lbls


def percentile_normalization(x, bottom=1, up=99.8, eps=1e-20, axis=(0, 1, 2)):
    percentiles = np.percentile(x, [bottom, up], keepdims=True, axis=axis)
    return (x - percentiles[0]) / (percentiles[1] - percentiles[0] + eps)


def create_patches(imgs, coms, lbls, patch_size):
    img_patches = []
    com_patches = []
    lbl_patches = []
    for img, com, lbl in zip(imgs, coms, lbls):
        slices = tuple([] for _ in range(lbl.ndim))
        for i in range(lbl.ndim):
            startpoints = np.arange(0, lbl.shape[i] - patch_size[i], patch_size[i] // 2)
            for start in startpoints:
                slices[i].append(slice(start, start + patch_size[i]))

        for sl in product(*slices):
            img_patches.append(img[(slice(None),) + sl])
            com_patches.append(com[(slice(None),) + sl])
            lbl_patches.append(lbl[sl])

    return (
        np.stack(img_patches, axis=0),
        np.stack(com_patches, axis=0),
        np.stack(lbl_patches, axis=0),
    )


def get_masks(lbls):
    mask = np.empty(lbls.shape, dtype=bool)
    for i in range(lbls.shape[0]):
        mask[i] = np.logical_and(
            lbls[i] >= 0,
            mask_bordering_lbls(lbls[i]),
        )
    return mask


class COMDataset(Dataset):
    def __init__(
        self,
        imgs: list[np.ndarray],
        lbls: list[np.ndarray],
        patch_size: tuple[int, ...],
        normalize: bool = True,
    ):
        if normalize:
            imgs = [percentile_normalization(img) for img in imgs]
        # TODO: allow for arbitrary input channels
        norm_imgs = [img[np.newaxis, ...] for img in imgs]

        com_lbls = [lbl_to_local_descriptors(lbl) for lbl in lbls]
        self.imgs, self.coms, patch_lbls = create_patches(
            norm_imgs, com_lbls, lbls, patch_size
        )
        self.masks = get_masks(patch_lbls)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.coms[idx], self.masks[idx]
