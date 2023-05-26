import fnmatch
import json
import numpy as np
import os
import tensorflow as tf
import time

import PIL
from PIL import Image
from PIL import ImageOps

from .utils import deltaT

AVAILABLE_AUGMENTATIONS = [
    "",
    "gray",
    "satur25",
    "satur125",
    "pixel-pepper-15",
    "pixel-pepper-30",
    "pixel-pepper-50",
    "pixel-rainbow-15",
    "pixel-rainbow-30",
    "pixel-rainbow-50",
    "square-black-patches-10-15px",
    "square-black-patches-20-15px",
    "square-rainbow-patches-10-15px",
    "square-rainbow-patches-20-15px",
    "upscale-2/3-bicubic",
    "upscale-2/3-box",
    "upscale-1/2-box",
    "upscale-1/5-box",

    "curtains-25",
    "curtains-50",
    "curtains-75",

    "gaussian-blur-1",
    "find-edges-1",
    "smudge-1",
    "emboss",

    ######################
    # "fliphoriz-rotate90",
    # "flipvert-rotate90",
    # "fliphoriz-rotate90-gray",
]


class RBCDiameterDataGen(tf.keras.utils.Sequence):
    def __init__(
        self,
        root_dir,
        image_width,
        yscale_factor,  # this should be set to image side
        batch_size=32,
        shuffle=True,
        max_count=0,
        inspection_path="",
        file_extensions=["jpg", "jpeg", "png"],
        augmentations=AVAILABLE_AUGMENTATIONS,
        cache_images_in_memory=False,
        verbose=True,
        # input_size=(256, 256, 3),
        exclusion_list_of_file_paths=[],
    ):
        self.root_dir = root_dir
        self.target_files = []
        self.batch_size = batch_size
        # self.input_size = input_size
        # self.image_width = input_size[0]
        self.image_width = image_width
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.file_extensions = file_extensions
        self.yscale_factor = yscale_factor
        self.verbose = verbose

        for root, _, filenames in os.walk(self.root_dir):
            for extension in self.file_extensions:
                for filename in fnmatch.filter(filenames, f"*.{extension}"):
                    self.target_files.append(os.path.join(root, filename))

        self.expected_answers = {}
        with open(f"{root_dir}/expected_answers.json") as f:
            self.expected_answers = json.load(f)

        # Exclusions
        exclusion_counter = 0
        for exs in exclusion_list_of_file_paths:
            if exs in self.target_files or self.expected_answers.get(exs.split("/")[1]):
                self.target_files.remove(exs)
                idx = exs
                if "/" in idx:
                    idx = idx.split("/")[-1]
                self.expected_answers.pop(idx)
                exclusion_counter += 1

        if exclusion_counter:
            print(f"{exclusion_counter} items Excluded")

        # Sanity checks
        assert len(self.target_files) > 0, "No files found"
        for trg_file in self.target_files:
            try:
                img = Image.open(trg_file)
            except Exception as e:
                if type(e) == PIL.UnidentifiedImageError:
                    os.remove(trg_file)
                    self.target_files.remove(trg_file)
                    self.expected_answers.pop(trg_file.split("/")[1])
                    print(f"cannot identify image file {trg_file}")
                    continue
                else:
                    raise (e)
            if img.width != img.height and (img.width < self.image_width or img.height < self.image_width):
                print(f"DELETING Image {trg_file} has non-square dimensions {img.width} x {img.height}")
                os.remove(trg_file)
                self.target_files.remove(trg_file)
                self.expected_answers.pop(trg_file.split("/")[1])
            elif img.width < self.image_width:
                raise Exception(f"Image {trg_file} width is too small")

        assert len(self.target_files) == len(self.expected_answers), "not all the expected files are available, we are in a pickle"
        print(f"Dataset with root '{root_dir}' has {len(self.target_files)} items")

        if shuffle:
            random_indexes = np.random.permutation(len(self.target_files))
            self.target_files = [self.target_files[ri] for ri in random_indexes[: max_count if max_count > 0 else len(self.target_files)]]

        if not shuffle and max_count > 0:
            self.target_files = self.target_files[0:max_count]

    def list_to_be_used_for_exclusion(self):
        return self.target_files

    def on_epoch_end(self):
        pass

    def __get_input(self, path, image_side):
        image = tf.keras.preprocessing.image.load_img(f"{path}")
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr, (image_side, image_side)).numpy()
        return image_arr / 255.0

    def __get_output(self, path):
        idx = path
        if "/" in idx:
            idx = idx.split("/")[-1]
        return self.expected_answers.get(idx)

    def __get_data(self, batches):
        # Generates data containing batch_size samples

        X_batch = np.asarray([self.__get_input(x, self.image_width) for x in batches])

        y0_batch = np.asarray([self.__get_output(y) for y in batches])
        y0_batch = y0_batch / self.yscale_factor

        return X_batch, y0_batch

    def get_batch(self, batch_index):
        t1 = time.time()
        batch_indexes = range(batch_index * self.batch_size, (batch_index + 1) * self.batch_size)
        items = []

        for index in batch_indexes:
            items.append(self[index])
        if self.verbose:
            deltaT(t1, "GB")
        return items

    def __getitem__(self, index):
        """
        __getitem__ method
        The role of __getitem__ method is to generate one batch of data. In this case, one batch of data will be (X, y) value pair where X represents the input and y represents the output.
        X will be a NumPy array of shape [batch_size, input_height, input_width, input_channel]. The image should be read from the disk, and the area of interest will be cropped out the image and preprocessing, if anything, has to be done according to the dataframe.
        y will be a tuple with two NumPy arrays of shape [batch_size, n_name] and [batch_size, n_type]). These will be one hot encoded value of the labels.
        """
        batches = self.target_files[index * self.batch_size : (index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return len(self.target_files) // self.batch_size
