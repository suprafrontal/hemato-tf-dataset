import os
import fnmatch
from random import random, shuffle
import tensorflow as tf
import numpy
from PIL import Image
import PIL
from PIL import ImageOps

# import cv2


def to_grayscale_then_rgb(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    return image


AVAILABLE_AUGMENTATIONS = [
    "",
    "gray",
    "rotate90",
    "rotate180",
    "fliphoriz",
    "flipvert",
    "satur25",
    "satur125",
    "fliphoriz-rotate90",
    "flipvert-rotate90",
    "fliphoriz-rotate90-gray",
    "flipvert-rotate90-gray",
    "fliphoriz-rotate180-gray",
    "flipvert-rotate180-gray",
    "gray-rotate90",
    "gray-rotate180",
    "gray-fliphoriz",
    "gray-flipvert",
    "fliphoriz-rotate90-satur25",
    "flipvert-rotate90-satur25",
    "satur25-rotate90",
    "satur25-fliphoriz",
    "fliphoriz-rotate90-satur125",
    "flipvert-rotate90-satur125",
    "satur125-rotate90",
    "satur125-fliphoriz",
]


class HemSelfSupDataset:
    # max_count = 0 -> no limit, otherwise cap the number of samples to max_count
    # augmentations is a list of augmentations to apply to each image
    # augmentations is a list of already permuted options like:
    # ['gray', 'rotate90', 'rotate180', 'fliphoriz', 'flipvert', 'gray-rotate90', 'gray-fliphoriz']
    # remove_multilabel_images
    # TRUE         if set to true removes any image that shows up in more than one category (like if the same image is under 1 and 2 or has RBC and NEUTRO)
    # 'one-sided'  removes any image that shows up in more than one category from all categories except 0
    # FALSE        do nothing
    # exclusive_rbc_dir if set to a value other than empty '', it is used as the sub directory where the RBC labeled images are AND all images in that directory that also have other labels will be excluded from training the dataset
    def __init__(
        self,
        root_dir,
        image_width,
        batch_size=32,
        shuffle=True,
        max_count=0,
        inspection_path="",
        file_extension="jpg",
        augmentations=AVAILABLE_AUGMENTATIONS,
        cache_images_in_memory=False,
    ):
        self.file_extension = file_extension
        self.index_map = []
        self.target_files = []
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.image_width = image_width
        self.inspection_path = inspection_path
        self.augmentations = augmentations
        self.enhance_for_purpule_stuff = False
        self.cache_images_in_memory = cache_images_in_memory

        for root, _, filenames in os.walk(self.root_dir):
            for filename in fnmatch.filter(filenames, f"*.{self.file_extension}"):
                self.target_files.append(os.path.join(root, filename))

        count_before = self.target_files.__len__()
        self.file_categories = {}
        offenders = []
        for trg_file in self.target_files:
            category = os.path.dirname(trg_file).split("/")[-1]
            fname = os.path.basename(trg_file)
            if not self.file_categories.get(fname):
                self.file_categories[fname] = [category]
            else:
                self.file_categories[fname].append(category)

        # Sanity checks
        assert len(self.target_files) > 0, "No files found"
        for trg_file in self.target_files:
            try:
                img = Image.open(trg_file)
            except Exception as e:
                if type(e) == PIL.UnidentifiedImageError:
                    os.remove(trg_file)
                    self.target_files.remove(trg_file)
                    print(f"cannot identify image file {trg_file}")
                    continue
                else:
                    print(f"Exception happened here and we need to raise it : {e}")
                    raise (e)
            if img.width != img.height and (
                img.width < self.image_width or img.height < self.image_width
            ):
                print(
                    f"DELETING Image {trg_file} has non-square dimensions {img.width} x {img.height}"
                )
                os.remove(trg_file)
                self.target_files.remove(trg_file)
            elif img.width < self.image_width:
                raise Exception(f"Image {trg_file} width is too small")

        self.cat_stats = {}
        self.ordered_cats = (
            []
        )  # order matters for index lookup, so we are sorting alphabetically
        for fname in self.file_categories:
            if len(self.file_categories[fname]) > 1:
                offenders.append(fname)

            for cat in self.file_categories[fname]:
                if not self.cat_stats.get(cat):
                    self.cat_stats[cat] = 1
                else:
                    self.cat_stats[cat] += 1
        self.ordered_cats = [x for x in self.cat_stats.keys()]
        self.ordered_cats.sort()

        print(
            "\x1b[1;31m ============================================================================================================== \x1b[0m"
        )
        print(
            " =============================================================================================================="
        )
        print(f"    Category stats: {self.cat_stats}")
        print(
            " =============================================================================================================="
        )
        print(
            "\x1b[1;31m ============================================================================================================== \x1b[0m"
        )
        self.one_hot_encoder = tf.keras.layers.CategoryEncoding(
            num_tokens=self.cat_stats.__len__(), output_mode="one_hot"
        )
        self.multi_hot_encoder = tf.keras.layers.CategoryEncoding(
            num_tokens=self.cat_stats.__len__(), output_mode="multi_hot"
        )

        # if shuffle:
        #     random_indexes = numpy.random.permutation(len(self.target_files))
        #     self.target_files = [
        #         self.target_files[ri]
        #         for ri in random_indexes[
        #             : max_count if max_count > 0 else len(self.target_files)
        #         ]
        #     ]

        if not shuffle and max_count > 0:
            self.target_files = self.target_files[0:max_count]

        # let's make sure adjacent items are not all same picture or same agumentation
        self.index_map = range(0, (len(self.augmentations) * len(self.target_files)))
        if shuffle:
            self.index_map = numpy.random.permutation(
                len(self.augmentations) * len(self.target_files)
            )

        self.mem_cache = [None] * (len(self.augmentations) * len(self.target_files))

    def __len__(self):
        return len(self.augmentations) * len(self.target_files)

    def get_batch(self, batch_index):
        batch_indexes = range(
            batch_index * self.batch_size, (batch_index + 1) * self.batch_size
        )
        items = []

        for index in batch_indexes:
            items.append(self[index])

        return items

    def __getitem__(self, index):
        if type(index) is int:
            ## ------ MEM CACHE
            index = index % self.__len__()
        if self.mem_cache[index]:
            return self.mem_cache[index]
        ## ----- END MEM CACHE
        elif type(index) is slice:
            r = []
            for idx in range(index.start, index.stop or len(self), index.step or 1):
                r.append(self.__getitem__(idx))
            return r
        elif type(index) is list:
            r = []
            for idx in index:
                r.append(self.__getitem__(idx))
            return r

        print(f">{index}<", end="")
        mapped_index = self.index_map[index]

        aug_idx = mapped_index % len(self.augmentations)
        file_idx = mapped_index // len(self.augmentations)

        trg_file = self.target_files[file_idx]
        identifier = ("/".join(trg_file.split("/")[-2:])).split(".")[0]

        img = Image.open(trg_file).convert("RGB")

        # if self.enhance_for_purpule_stuff:
        #     img = ImageOps.autocontrast(img)
        #     img = ImageOps.equalize(img)
        #     im = cv2.imread(trg_file)
        #     img_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        #     hsv_color1 = numpy.asarray([106 / 2, 45, 80])  # dark
        #     hsv_color2 = numpy.asarray([325 / 2, 255, 255])  # bright
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
        #     img = numpy.array(final, dtype="float32")
        if False:
            pass
        else:
            if "gray" in self.augmentations[aug_idx]:
                converter = PIL.ImageEnhance.Color(img)
                img = converter.enhance(0.0)
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
                numpy.array(img),
                [self.image_width, self.image_width],
                method=tf.image.ResizeMethod.BICUBIC,
                preserve_aspect_ratio=False,
            )  # we can ignore aspect ratio, because above here we crop a square
            # img = tf.image.crop_and_resize([numpy.array(img)], [[0,0,self.image_width,self.image_width]], [0], [self.image_width, self.image_width], method=tf.image.ResizeMethod.BILINEAR)[0]
            # img = tf.image.per_image_standardization(img)
            img = numpy.array(img, dtype="float32")

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

        item = {
            "img": img,
            "augmentations": self.augmentations[aug_idx],
            "identifier": f"{identifier}-{self.augmentations[aug_idx]}",
            "target_obj_file_path": trg_file,
        }

        if self.cache_images_in_memory:
            self.mem_cache[index] = item

        return item
