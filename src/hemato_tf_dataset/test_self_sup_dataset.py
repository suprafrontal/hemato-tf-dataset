import numpy
import math
from hemato_tf_dataset import HemSelfSupDataset


def test_HemSelfSupDataset():
    ds = HemSelfSupDataset("tests/test_data", image_width=256, augmentations=[""])
    assert ds
    assert ds.__len__() == 3
    x1 = ds.get_batch(0)
    assert len(x1) == ds.batch_size
    item = ds[0]
    assert math.isclose(numpy.max(item["img"]) , 1.0)


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


def test_pixel_rainbow_50():
    ds = HemSelfSupDataset("tests/test_data", image_width=256, augmentations=["pixel-rainbow-50"])
    assert ds
    assert ds[0]


def test_max_count():
    ds = HemSelfSupDataset("tests/test_data", image_width=256)
    assert ds
    assert ds.__len__() == 81

    dsmax = HemSelfSupDataset("tests/test_data", image_width=256, max_count=1)
    assert dsmax
    assert dsmax.__len__() == 27


def test_square_black_patches():
    ds = HemSelfSupDataset("tests/test_data", image_width=256, augmentations=["square-black-patches-10-15px"])
    assert ds
    assert ds[0]


def test_upscale_1_5_box():
    ds = HemSelfSupDataset("tests/test_data", image_width=256, augmentations=["upscale-1/5-box"])
    assert ds
    assert ds[0]


def test_square_rainbow_patches():
    ds = HemSelfSupDataset("tests/test_data", image_width=256, augmentations=["square-rainbow-patches-13-25px"])
    assert ds
    assert ds[0]


def test_shuffle_4x4():
    ds = HemSelfSupDataset("tests/test_data", image_width=256, augmentations=["shuffle-4x4"])
    assert ds
    assert ds[0]


def test_invert():
    ds = HemSelfSupDataset("tests/test_data", image_width=256, augmentations=["invert"])
    assert ds
    assert ds[0]

def test_curtain():
    ds = HemSelfSupDataset("tests/test_data", image_width=256, augmentations=["curtain-25"])
    assert ds
    assert ds[0]
    ds = HemSelfSupDataset("tests/test_data", image_width=256, augmentations=["curtain-50"])
    assert ds
    assert ds[0]
    ds = HemSelfSupDataset("tests/test_data", image_width=256, augmentations=["curtain-75"])
    assert ds
    assert ds[0]
