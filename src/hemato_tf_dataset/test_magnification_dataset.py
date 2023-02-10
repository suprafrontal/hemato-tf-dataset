import numpy
from hemato_tf_dataset import RBCDiameterDataGen


def test_RBCDiamterDataset():
    ds = RBCDiameterDataGen("tests/magnification_test_data", image_width=256, augmentations=[""])
    assert ds != None
    # assert ds.__len__() == 4
    # x1 = ds.get_batch(0)
    # assert len(x1) == ds.batch_size
    # item = ds[0]
    # assert numpy.max(item["img"]) == 1.0
