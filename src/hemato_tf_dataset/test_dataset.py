from .dataset import version, image_distances
import numpy as np
import math

def test_version():
    assert version()


def test_images_distances_1():
    a = np.ones((7, 10, 10, 3))
    b = np.zeros((7, 10, 10, 3))
    deltas = image_distances(a, b)
    assert all([(d == 1) for d in deltas])


def test_images_distances_2():
    c = 0.6
    a = np.ones((7, 10, 10, 3))
    a *= c
    b = np.zeros((7, 10, 10, 3))
    deltas = image_distances(a, b)
    np.testing.assert_almost_equal(deltas, np.repeat([c], 7))


def test_images_distances_3():
    a = [
        [  # image 1
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.6, 0.7, 0.8],
            ],  #
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.6, 0.7, 0.8],
            ],  #
        ],
    ]
    b = np.zeros((1, 2, 4, 3))
    deltas = image_distances(a, b)
    np.testing.assert_almost_equal(deltas, np.repeat([(2.0 + 2.0 + 2.0 + (0.6 + 0.7 + 0.8) + (0.6 + 0.7 + 0.8)) / (2.0 * 3 * 4)], 1))

def test_images_distances_4():
    a = np.ones((7, 10, 10, 3))
    b = np.zeros((7, 10, 10, 3))
    deltas = image_distances(a, b, channel_wights=[0.0, 0.0, 0.0])
    assert all([(d == 0) for d in deltas])

    deltas = image_distances(a, b, channel_wights=[0.0, 1.0, 0.0])
    assert all([(math.isclose(d, 1.)) for d in deltas])

    deltas = image_distances(a, b, channel_wights=[0.0, 1.0, 1.0])
    assert all([(math.isclose(d, 1.)) for d in deltas])

    deltas = image_distances(a, b, channel_wights=[0.0, 0.27, 0.0])
    assert all([(math.isclose(d, 0.27)) for d in deltas])

    deltas = image_distances(a, b, channel_wights=[0.0, .6, .09])
    assert all([(math.isclose(d, 0.345)) for d in deltas])

    deltas = image_distances(a, b, channel_wights=[0.1, .5, .09])
    assert all([(math.isclose(d, 0.23)) for d in deltas])

def test_images_distances_5():
    c = 0.6
    a = np.ones((7, 10, 10, 3))
    a *= c
    b = np.zeros((7, 10, 10, 3))
    deltas = image_distances(a, b, [1.0, 1.0, 1.0])
    np.testing.assert_almost_equal(deltas, np.repeat([c], 7))
    deltas = image_distances(a, b, [.0, .0, .0])
    np.testing.assert_almost_equal(deltas, np.repeat([0], 7))
    deltas = image_distances(a, b, [1.0, .0, .0])
    np.testing.assert_almost_equal(deltas, np.repeat([c*1.], 7))


def test_images_distances_6():
    a = [
        [  # image 1
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.6, 0.7, 0.8],
            ],  #
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.6, 0.7, 0.8],
            ],  #
        ],
    ]
    b = np.zeros((1, 2, 4, 3))
    deltas = image_distances(a, b)
    np.testing.assert_almost_equal(deltas, np.repeat([(2.0 + 2.0 + 2.0 + (0.6 + 0.7 + 0.8) + (0.6 + 0.7 + 0.8)) / (2.0 * 3 * 4)], 1))
    deltas = image_distances(a, b, [1.0, 1.0, 1.0])
    np.testing.assert_almost_equal(deltas, np.repeat([(2.0 + 2.0 + 2.0 + (0.6 + 0.7 + 0.8) + (0.6 + 0.7 + 0.8)) / (2.0 * 3 * 4)], 1))
    deltas = image_distances(a, b, [.0, .0, .0])
    np.testing.assert_almost_equal(deltas, np.repeat([0], 1))
    deltas = image_distances(a, b, [1.0, .0, .0])
    np.testing.assert_almost_equal(deltas, np.repeat([0.525], 1))

