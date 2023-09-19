import math
import numpy
from hemato_tf_dataset import HemBinaryClassificationDataset


def test_HemBinaryClassificationDataset():
    ds = HemBinaryClassificationDataset("tests/binary_classification_test_data",
                                        image_width=256,
                                        cache_images_in_memory=False,
                                        augmentations=    ["NO_AUGMENTATION",    "gray",    "invert"],
                                        shuffle=True,)
    assert ds is not None
    assert ds.__len__() == 224 * 3 # number of augmentations
    x1 = ds.get_batch(0)
    assert len(x1) == ds.batch_size
    item = ds[0]
    assert math.isclose(numpy.max(item["img"]), 1.0)
    assert math.isclose(numpy.max(item["img_tensor"]), 1.0)
    assert item["zero_v_one"] in [0, 1]
    assert item["hot_one"].numpy() is not None

    item = ds[1]
    assert math.isclose(numpy.max(item["img"]), 1.0)
    assert math.isclose(numpy.max(item["img_tensor"]), 1.0)
    assert item["zero_v_one"] in [0, 1]
    assert item["hot_one"].numpy() is not None

    for i in range(0, 1000):
        item = ds[i]
        if "HasTheThing" in item["target_obj_file_path"]:
            assert item["zero_v_one"] == 0
            assert all(item["hot_one"].numpy() == [1, 0])
        else:
            assert item["zero_v_one"] == 1
            assert all(item["hot_one"].numpy() == [0, 1])