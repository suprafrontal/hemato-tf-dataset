import numpy
from hemato_tf_dataset import HemSelfSupDataset


def test_HemSelfSupDataset():
    ds = HemSelfSupDataset("tests/test_data", image_width=256, augmentations=[""])
    assert ds
    assert ds.__len__() == 3
    x1 = ds.get_batch(0)
    assert len(x1) == ds.batch_size
    item = ds[0]
    assert numpy.max(item["img"]) == 1.0


def test_batch_get():
    ds = HemSelfSupDataset("tests/test_data", image_width=256, augmentations=[""], cache_images_in_memory=True)
    assert ds
    assert ds.__len__() == 3
    b1 = ds.get_batch(0)
    assert len(b1) == ds.batch_size


def test_pixel_noise_15():
    ds = HemSelfSupDataset("tests/test_data", image_width=256, augmentations=["pixel-noise-15"])
    assert ds
    assert ds[0]
