import math
import numpy
from hemato_tf_dataset import HemBinaryClassificationDataset


def test_HemBinaryClassificationDataset():
    ds = HemBinaryClassificationDataset("tests/binary_classification_test_data", image_width=256, cache_images_in_memory=True)
    assert ds is not None
    assert ds.__len__() == 54
    x1 = ds.get_batch(0)
    assert len(x1) == ds.batch_size
    item = ds[0]
    assert math.isclose(numpy.max(item["img"]), 1.0)
    assert math.isclose(numpy.max(item["img_tensor"]), 1.0)
    assert item["zero_v_one"] in [0, 1]
    assert item["hot_one"].numpy() is not None
