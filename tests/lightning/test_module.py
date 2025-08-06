import numpy as np
from torch import utils
from lightning import Trainer
from com_iseg.lightning import module
from com_iseg.data import dataset


def test_trainer_fast_dev_run():
    model = module.COMModule(depth=3)
    size = 30
    lbl = np.zeros((2, size, size, size), dtype=np.int8)

    lbl[0, :4, :4, 10:14] = 1
    lbl[0, 1:4, 7:8, 10:14] = 2
    lbl[0, 11:14, 17:18, 10:14] = -3
    lbl[0, 21:, 21:, 10:14] = 4  # skip lbl = 1 to trigger 'if obj is None'

    lbl[1, 4:8, 10:15, 20:25] = -1

    lbls = [lbl[i] for i in range(2)]

    patch_size = (16, 16, 16)
    ds = dataset.COMDataset(
        [lbl.astype(float) for lbl in lbls],
        lbls,
        patch_size,
        normalize=False,
    )
    loader = utils.data.DataLoader(ds)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model=model, train_dataloaders=loader)


def test_trainer_predict():
    model = module.COMModule(depth=3)
    size = 30
    lbl = np.zeros((2, size, size, size), dtype=np.int8)

    lbl[0, :4, :4, 10:14] = 1
    lbl[0, 1:4, 7:8, 10:14] = 2
    lbl[0, 11:14, 17:18, 10:14] = -3
    lbl[0, 21:, 21:, 10:14] = 4  # skip lbl = 1 to trigger 'if obj is None'

    lbl[1, 4:8, 10:15, 20:25] = -1

    lbls = [lbl[i] for i in range(2)]

    patch_size = (16, 16, 16)
    ds = dataset.COMDataset(
        [lbl.astype(float) for lbl in lbls],
        lbls,
        patch_size,
        normalize=False,
    )
    loader = utils.data.DataLoader([d[0] for d in ds])
    trainer = Trainer()
    predictions = trainer.predict(model=model, dataloaders=loader)
    for prediction in predictions:
        assert len(prediction) == 2
        assert (
            prediction[0].shape
            == (
                1,
                1,
            )
            + patch_size
        )
        assert (
            prediction[1].shape
            == (
                1,
                3,
            )
            + patch_size
        )
        # assert (
        #     prediction[2].shape
        #     == (
        #         1,
        #         3,
        #     )
        #     + patch_size
        # )
        # assert (
        #     prediction[3].shape
        #     == (
        #         1,
        #         3,
        #     )
        #     + patch_size
        # )
