import copy
import math
import fnmatch
import json
from random import random, randint
import numpy as np
import os
import tensorflow as tf
import time

import PIL
from PIL import Image
from PIL import ImageOps, ImageFilter

from .utils import deltaT, translate_wrap

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


class CellDetectionDataset:
    """
    yscale_factor should be manually set to the "image height"
    """

    def __init__(
        self,
        root_dir,
        path_to_expected_answers_json,
        image_width,
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
        yscale_factor=0,
    ):
        self.root_dir = root_dir
        self.index_map = []
        self.target_files = []
        self.batch_size = batch_size
        # self.input_size = input_size
        # self.image_width = input_size[0]
        self.inspection_path = inspection_path
        self.image_width = image_width
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.file_extensions = file_extensions
        self.yscale_factor = yscale_factor
        self.cache_images_in_memory = cache_images_in_memory
        self.verbose = verbose

        for root, _, filenames in os.walk(self.root_dir):
            for extension in self.file_extensions:
                for filename in fnmatch.filter(filenames, f"*.{extension}"):
                    self.target_files.append(os.path.join(root, filename))

        self.expected_answers = {}
        with open(path_to_expected_answers_json) as f:
            self.expected_answers = json.load(f)

        # Exclusions
        exclusion_counter = 0
        for exs in exclusion_list_of_file_paths:
            if exs in self.target_files or self.expected_answers.get(exs.split("/")[1]):
                self.target_files.remove(exs)
                self.expected_answers.pop(os.path.basename(exs))
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

        if max_count < 1:
            max_count = len(self.target_files)

        # let's make sure adjacent items are not all same picture or same agumentation
        self.index_map = range(0, (len(self.augmentations) * min(max_count, len(self.target_files))))
        if shuffle:
            self.index_map = np.random.permutation(len(self.augmentations) * min(max_count, len(self.target_files)))

        self.mem_cache = [None] * (len(self.augmentations) * len(self.target_files))

    def __len__(self):
        return len(self.index_map)

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
        return self.expected_answers.get(os.path.basename(path))

    def __get_data(self, batches):
        # Generates data containing batch_size samples

        X_batch = np.asarray([self.__get_input(x, self.image_width) for x in batches])
        y0_batch = [np.asarray(self.__get_output(y)) / self.yscale_factor for y in batches]

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
        # """
        # __getitem__ method
        # The role of __getitem__ method is to generate one batch of data. In this case, one batch of data will be (X, y) value pair where X represents the input and y represents the output.
        # X will be a NumPy array of shape [batch_size, input_height, input_width, input_channel]. The image should be read from the disk, and the area of interest will be cropped out the image and preprocessing, if anything, has to be done according to the dataframe.
        # y will be a tuple with two NumPy arrays of shape [batch_size, n_name] and [batch_size, n_type]). These will be one hot encoded value of the labels.
        # """
        # batches = self.target_files[index * self.batch_size : (index + 1) * self.batch_size]
        # X, y = self.__get_data(batches)
        # return X, y
        t1 = time.time()
        if type(index) is int:
            ## ------ MEM CACHE
            index = index % self.__len__()
        if self.mem_cache[index]:
            if self.verbose:
                deltaT(t1, "GIc")
            return self.mem_cache[index]
        ## ----- END MEM CACHE
        elif type(index) is slice:
            r = []
            for idx in range(index.start, index.stop or len(self), index.step or 1):
                r.append(self.__getitem__(idx))
            if self.verbose:
                deltaT(t1, "GIc")
            return r
        elif type(index) is list:
            r = []
            for idx in index:
                r.append(self.__getitem__(idx))
            if self.verbose:
                deltaT(t1, "GIc")
            return r
        elif type(index) is tuple:
            r = []
            for idx in index:
                r.append(self.__getitem__(idx))
            if self.verbose:
                deltaT(t1, "GIc")
            return r

        if self.verbose:
            print(f">{index}<", end="")
        mapped_index = self.index_map[index]

        aug_idx = mapped_index % len(self.augmentations)
        file_idx = mapped_index // len(self.augmentations)

        trg_file = self.target_files[file_idx]
        identifier = ("/".join(trg_file.split("/")[-2:])).split(".")[0]

        img = Image.open(trg_file).convert("RGB")
        rects = self.expected_answers.get(os.path.basename(trg_file))

        og_img = copy.deepcopy(img)

        # if self.enhance_for_purpule_stuff:
        #     img = ImageOps.autocontrast(img)
        #     img = ImageOps.equalize(img)
        #     im = cv2.imread(trg_file)
        #     img_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        #     hsv_color1 = np.asarray([106 / 2, 45, 80])  # dark
        #     hsv_color2 = np.asarray([325 / 2, 255, 255])  # bright
        #     cv2.resize(
        #         img_hsv, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC
        #     )
        #     cv2.resize(img_hsv, (0, 0), fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
        #     img_blues = cv2.inRange(img_hsv, hsv_color1, hsv_color2)
        #     gray_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #     gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        #     fg = cv2.bitwise_or(im, im, mask=img_blues)
        #     mask_inv = cv2.bitwise_not(img_blues)
        #     fg_back_inv = cv2.bitwise_or(gray_img, gray_img, mask=mask_inv)
        #     final = cv2.bitwise_or(fg, fg_back_inv)
        #     cv2.resize(final, (self.image_width, self.image_width))
        #     img = np.array(final, dtype="float32")
        if False:
            pass
        else:
            if "gray" in self.augmentations[aug_idx]:
                converter = PIL.ImageEnhance.Color(img)
                img = converter.enhance(0.0)
            if "invert" in self.augmentations[aug_idx]:
                img = PIL.ImageOps.invert(img)
            if "rotate90" in self.augmentations[aug_idx]:
                img = img.rotate(90)
            if "rotate180" in self.augmentations[aug_idx]:
                img = img.rotate(180)
            if "fliphoriz" in self.augmentations[aug_idx]:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if "flipvert" in self.augmentations[aug_idx]:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if "satur25" in self.augmentations[aug_idx]:
                converter = PIL.ImageEnhance.Color(img)
                img = converter.enhance(0.25)
            if "satur125" in self.augmentations[aug_idx]:
                converter = PIL.ImageEnhance.Color(img)
                img = converter.enhance(1.25)
            if "pixel-pepper-" in self.augmentations[aug_idx] or "pixel-rainbow-" in self.augmentations[aug_idx]:
                is_rainbow = "rainbow" in self.augmentations[aug_idx]
                noise_amount = abs(float(self.augmentations[aug_idx][13:]) / 100.0)  # ABS because pepper and rainbow have different lenght and - creeps in
                np_img = np.array(img)
                mask = np.random.rand(img.height, img.width) < noise_amount
                for idx in np.ndindex(img.height - 1, img.width - 1):
                    if mask[idx] == True:
                        if is_rainbow:
                            np_img[idx[0], idx[1], :] = np.array((randint(0, 255), randint(0, 255), randint(0, 255)), dtype="uint8")
                        else:  # just pepper
                            np_img[idx[0], idx[1], :] = np.array((0, 0, 0), dtype="uint8")
                    else:
                        pass
                img = Image.fromarray(np_img)
            if "square-black-patches-" in self.augmentations[aug_idx] or "square-rainbow-patches-" in self.augmentations[aug_idx]:
                is_rainbow = "rainbow" in self.augmentations[aug_idx]
                patch_count = int(self.augmentations[aug_idx].split("-")[-2])
                patch_side = int(self.augmentations[aug_idx].split("-")[-1][:-2])
                np_img = np.array(img)
                mask = np.zeros((img.height, img.width))
                for _ in range(0, patch_count):
                    px = randint(0, 255 - patch_side)
                    py = randint(0, 255 - patch_side)
                    mask[py : py + patch_side, px : px + patch_side] = 1
                for idx in np.ndindex(img.height - 1, img.width - 1):
                    if mask[idx] == True:
                        if is_rainbow:
                            np_img[idx[0], idx[1], :] = np.array((randint(0, 255), randint(0, 255), randint(0, 255)), dtype="uint8")
                        else:  # just pepper
                            np_img[idx[0], idx[1], :] = np.array((0, 0, 0), dtype="uint8")
                    else:
                        pass
                img = Image.fromarray(np_img)
            if "shuffle-4x4" in self.augmentations[aug_idx]:
                np_img = np.array(img)
                new_img = np.zeros(np_img.shape)
                (wimg, himg, _) = np_img.shape
                wimg = int(wimg / 4)
                himg = int(himg / 4)
                # print(f"{new_img.shape} - {np_img.shape} - {wimg} {himg}")
                new_square_locations = np.random.permutation(16)  # 4x4

                for idx, loc in enumerate(new_square_locations):
                    # a = [(loc / 4) * wimg: (loc / 4) * (wimg+1), (loc % 4) * himg:(loc %4 ) * (himg +1)]
                    # b = [(idx / 4) * wimg: (idx / 4) * (wimg+1), (idx % 4) * himg:(idx %4 ) * (himg +1)]
                    # print(
                    #     f"""\n{idx} -> {loc} |
                    #     {int(math.floor(loc / 4) * wimg)} : {int((math.floor(loc / 4) + 1) * wimg)}, {int(math.floor(loc % 4) * himg)} : {int((math.floor(loc % 4) + 1) * himg)} ||||||
                    #     {int(math.floor(idx / 4) * wimg)} : {int((math.floor(idx / 4) + 1) * wimg)}, {int(math.floor(idx % 4) * himg)} : {int((math.floor(idx % 4) + 1) * himg)}"""
                    # )
                    # print(
                    #     f"{int((loc / 4) * wimg) - int((loc / 4 + 1) * wimg)} {int((loc % 4) * himg) - int((loc % 4 + 1) * himg)} {int((idx / 4) * wimg) - int((idx / 4 + 1) * wimg)} {int((idx % 4) * himg) - int((idx % 4 + 1) * himg)}"
                    # )
                    new_img[
                        int(math.floor(loc / 4) * wimg) : int(min(wimg * 4, (math.floor(loc / 4) + 1) * wimg)),
                        int(math.floor(loc % 4) * himg) : int(min(himg * 4, (math.floor(loc % 4) + 1) * himg)),
                        :,
                    ] = np_img[
                        int(math.floor(idx / 4) * wimg) : int(min(wimg * 4, (math.floor(idx / 4) + 1) * wimg)),
                        int(math.floor(idx % 4) * himg) : int(min(himg * 4, (math.floor(idx % 4) + 1) * himg)),
                        :,
                    ]
                img = Image.fromarray(new_img.astype("uint8"))
            if "upscale-2/3-bicubic" in self.augmentations[aug_idx]:
                new_img = img.resize((int(img.size[0] * 2 / 3), int(img.size[1] * 2 / 3)))
                new_img = new_img.resize((img.size[0], img.size[1]), resample=Image.BICUBIC)
                img = new_img
            if "upscale-2/3-box" in self.augmentations[aug_idx]:
                new_img = img.resize((int(img.size[0] * 2 / 3), int(img.size[1] * 2 / 3)))
                new_img = new_img.resize((img.size[0], img.size[1]), resample=Image.BOX)
                img = new_img
            if "upscale-1/2-box" in self.augmentations[aug_idx]:
                new_img = img.resize((int(img.size[0] * 1 / 2), int(img.size[1] * 1 / 2)))
                new_img = new_img.resize((img.size[0], img.size[1]), resample=Image.BOX)
                img = new_img
            if "upscale-1/5-box" in self.augmentations[aug_idx]:
                new_img = img.resize((int(img.size[0] * 1 / 5), int(img.size[1] * 1 / 5)))
                new_img = new_img.resize((img.size[0], img.size[1]), resample=Image.BOX)
                img = new_img

            if "gaussian-blur-" in self.augmentations[aug_idx]:
                sigma = float(self.augmentations[aug_idx].split("-")[-1])
                img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
            if "find-edges" in self.augmentations[aug_idx]:
                img = img.filter(ImageFilter.FIND_EDGES)
            if "smudge-1" in self.augmentations[aug_idx]:
                img = img.filter(ImageFilter.SMOOTH_MORE)
            if "emboss" in self.augmentations[aug_idx]:
                img = img.filter(ImageFilter.EMBOSS)
            if "curtain" in self.augmentations[aug_idx]:
                shift = float(self.augmentations[aug_idx].split("-")[-1])
                w, _ = img.size
                shift = int(w * shift / 100)
                img = translate_wrap(img, shift, shift)
            #################################################################################################################################################################################
            side = min(img.width, img.height)
            img = img.crop((0, 0, side, side))

            if self.inspection_path:
                img.save(
                    os.path.join(
                        self.inspection_path,
                        f"{identifier}-{self.augmentations[aug_idx]}.jpg",
                    )
                )

            # *   `tf.image.adjust_brightness`
            # *   `tf.image.adjust_contrast`
            # *   `tf.image.adjust_gamma`
            # *   `tf.image.adjust_hue`
            # *   `tf.image.adjust_jpeg_quality`
            # *   `tf.image.adjust_saturation`
            # *   `tf.image.random_brightness`
            # *   `tf.image.random_contrast`
            # *   `tf.image.random_hue`
            # *   `tf.image.random_saturation`
            # *   `tf.image.per_image_standardization`

            img = tf.image.resize(
                np.array(img),
                [self.image_width, self.image_width],
                method=tf.image.ResizeMethod.BICUBIC,
                preserve_aspect_ratio=False,
            )  # we can ignore aspect ratio, because above here we crop a square
            # img = tf.image.crop_and_resize([np.array(img)], [[0,0,self.image_width,self.image_width]], [0], [self.image_width, self.image_width], method=tf.image.ResizeMethod.BILINEAR)[0]
            # img = tf.image.per_image_standardization(img)
            img = np.array(img, dtype="float32")
            og_img = np.array(og_img, dtype="float32")

        imw, imh, imc = img.shape
        if imw != self.image_width or imw != imh or imc != 3:
            img = img[0 : self.image_width, 0 : self.image_width, 0:3]

        # single_category = int(os.path.dirname(trg_file).split('/')[-1])
        # single_category = os.path.dirname(trg_file).split("/")[-1]
        # category_index = self.ordered_cats.index(single_category)
        # fname = os.path.basename(trg_file)
        # multi_category = self.file_categories[fname]
        # if self.multi_label:
        #     hot_one = self.multi_hot_encoder([int(ml) for ml in multi_category])
        # else:
        #     hot_one = self.one_hot_encoder(category_index)

        # normalize
        img *= 1.0 / img.max()
        og_img *= 1.0 / og_img.max()

        item = {
            "img": img,
            # "og_img": og_img,
            "augmentations": self.augmentations[aug_idx],
            "identifier": f"{identifier}-{self.augmentations[aug_idx]}",
            "target_obj_file_path": trg_file,
            "target_rects": rects,
            "aug_idx": aug_idx,
            "augmentations": self.augmentations[aug_idx],
        }

        if self.cache_images_in_memory:
            self.mem_cache[index] = item

        if self.verbose:
            deltaT(t1, "GI")

        return item

    def __len__(self):
        return len(self.target_files)  # // self.batch_size
