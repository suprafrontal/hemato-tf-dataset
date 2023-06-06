# import tomllib wait until python 3.11
import numpy as np
import numpy.typing as npt


def version():
    # wait until python 3.11
    # with open("pyproject.toml", "rb") as f:
    #     data = tomllib.load(f)
    #     print(data["tool.poetry"])
    with open("pyproject.toml", encoding="utf-8") as f:
        read_data = f.readline()
        read_data = f.readline()
        read_data = f.readline()
        return read_data.split(" ")[2][1:-2]


VERSION = version()


class Dataset:
    def version(self):
        return VERSION


def image_distances(a: npt.ArrayLike, b: npt.ArrayLike, channel_wights: npt.ArrayLike = [1.0, 1.0, 1.0]):
    """ "
    a and b are arrays of RGB images
    """
    assert len(a) == len(b)
    assert len(channel_wights) == 3

    a = np.array(a)
    b = np.array(b)
    # absolute_delta = np.sum(np.sum(np.sum([abs(a[i] - b[i]) for i in range(len(a))], axis=3), axis=2), axis=1)
    absolute_deltas = [np.sum(((abs(b[i] - a[i])) * channel_wights)) for i in range(len(a))]
    white_image = np.ones(a[0].shape)
    max_possible_delta = np.sum(white_image*np.ceil(channel_wights))
    if max_possible_delta == 0:
        return np.zeros(len(a))
    return absolute_deltas / max_possible_delta
