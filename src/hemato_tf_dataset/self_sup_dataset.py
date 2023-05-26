import copy
import os
import fnmatch
import math
import time
from random import random, randint
import tensorflow as tf
import numpy
from PIL import Image
import PIL
from PIL import ImageOps
from PIL import ImageFile
from PIL import ImageFilter

from .utils import deltaT, translate_wrap

# import cv2


def to_grayscale_then_rgb(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    return image


AVAILABLE_AUGMENTATIONS = [
    "",
    "gray",
    "invert",
    "satur25",
    "satur125",
    "pixel-pepper-15",
    "pixel-pepper-30",
    "pixel-pepper-50",
    "pixel-rainbow-15",
    "pixel-rainbow-30",
    "pixel-rainbow-50",

    "curtains-25",
    "curtains-50",
    "curtains-75",

    "gaussian-blur-1",
    "find-edges-1",
    "smudge-1",
    "emboss",

    "square-black-patches-10-15px",
    "square-black-patches-20-15px",
    "square-rainbow-patches-10-15px",
    "square-rainbow-patches-20-15px",
    "shuffle-4x4",
    "upscale-2/3-bicubic",
    "upscale-2/3-box",
    "upscale-1/2-box",
    "upscale-1/5-box",
    ######################
    # "fliphoriz-rotate90",
    # "flipvert-rotate90",
    # "fliphoriz-rotate90-gray",
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
        file_extensions=["jpg", "jpeg", "png"],
        augmentations=AVAILABLE_AUGMENTATIONS,
        cache_images_in_memory=False,
        verbose=True,
        ignore_truncated_image_errors=True,
    ):
        self.file_extensions = file_extensions
        self.index_map = []
        self.target_files = []
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.image_width = image_width
        self.inspection_path = inspection_path
        self.augmentations = augmentations
        # self.enhance_for_purpule_stuff = False
        self.cache_images_in_memory = cache_images_in_memory
        self.verbose = verbose

        if ignore_truncated_image_errors:
            ImageFile.LOAD_TRUNCATED_IMAGES = True

        for root, _, filenames in os.walk(self.root_dir):
            for extension in self.file_extensions:
                for filename in fnmatch.filter(filenames, f"*.{extension}"):
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
            if img.width != img.height and (img.width < self.image_width or img.height < self.image_width):
                print(f"DELETING Image {trg_file} has non-square dimensions {img.width} x {img.height}")
                os.remove(trg_file)
                self.target_files.remove(trg_file)
            elif img.width < self.image_width:
                raise Exception(f"Image {trg_file} width is too small")

        self.cat_stats = {}
        self.ordered_cats = []  # order matters for index lookup, so we are sorting alphabetically
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

        print("\x1b[1;31m ============================================================================================================== \x1b[0m")
        print(" ==============================================================================================================")
        print(f"    Category stats: {self.cat_stats}")
        print(" ==============================================================================================================")
        print("\x1b[1;31m ============================================================================================================== \x1b[0m")
        self.one_hot_encoder = tf.keras.layers.CategoryEncoding(num_tokens=self.cat_stats.__len__(), output_mode="one_hot")
        self.multi_hot_encoder = tf.keras.layers.CategoryEncoding(num_tokens=self.cat_stats.__len__(), output_mode="multi_hot")

        if not shuffle and max_count > 0:
            self.target_files = self.target_files[0:max_count]

        if max_count < 1:
            max_count = len(self.target_files)

        # let's make sure adjacent items are not all same picture or same agumentation
        self.index_map = range(0, (len(self.augmentations) * min(max_count, len(self.target_files))))
        if shuffle:
            self.index_map = numpy.random.permutation(len(self.augmentations) * min(max_count, len(self.target_files)))

        self.mem_cache = [None] * (len(self.augmentations) * len(self.target_files))

    def __len__(self):
        return len(self.index_map)

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
        og_img = copy.deepcopy(img)

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
                np_img = numpy.array(img)
                mask = numpy.random.rand(img.height, img.width) < noise_amount
                for idx in numpy.ndindex(img.height - 1, img.width - 1):
                    if mask[idx] == True:
                        if is_rainbow:
                            np_img[idx[0], idx[1], :] = numpy.array((randint(0, 255), randint(0, 255), randint(0, 255)), dtype="uint8")
                        else:  # just pepper
                            np_img[idx[0], idx[1], :] = numpy.array((0, 0, 0), dtype="uint8")
                    else:
                        pass
                img = Image.fromarray(np_img)
            if "square-black-patches-" in self.augmentations[aug_idx] or "square-rainbow-patches-" in self.augmentations[aug_idx]:
                is_rainbow = "rainbow" in self.augmentations[aug_idx]
                patch_count = int(self.augmentations[aug_idx].split("-")[-2])
                patch_side = int(self.augmentations[aug_idx].split("-")[-1][:-2])
                np_img = numpy.array(img)
                mask = numpy.zeros((img.height, img.width))
                for _ in range(0, patch_count):
                    px = randint(0, 255 - patch_side)
                    py = randint(0, 255 - patch_side)
                    mask[py : py + patch_side, px : px + patch_side] = 1
                for idx in numpy.ndindex(img.height - 1, img.width - 1):
                    if mask[idx] == True:
                        if is_rainbow:
                            np_img[idx[0], idx[1], :] = numpy.array((randint(0, 255), randint(0, 255), randint(0, 255)), dtype="uint8")
                        else:  # just pepper
                            np_img[idx[0], idx[1], :] = numpy.array((0, 0, 0), dtype="uint8")
                    else:
                        pass
                img = Image.fromarray(np_img)
            if "shuffle-4x4" in self.augmentations[aug_idx]:
                np_img = numpy.array(img)
                new_img = numpy.zeros(np_img.shape)
                (wimg, himg, _) = np_img.shape
                wimg = int(wimg / 4)
                himg = int(himg / 4)
                # print(f"{new_img.shape} - {np_img.shape} - {wimg} {himg}")
                new_square_locations = numpy.random.permutation(16)  # 4x4

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
                numpy.array(img),
                [self.image_width, self.image_width],
                method=tf.image.ResizeMethod.BICUBIC,
                preserve_aspect_ratio=False,
            )  # we can ignore aspect ratio, because above here we crop a square
            # img = tf.image.crop_and_resize([numpy.array(img)], [[0,0,self.image_width,self.image_width]], [0], [self.image_width, self.image_width], method=tf.image.ResizeMethod.BILINEAR)[0]
            # img = tf.image.per_image_standardization(img)
            img = numpy.array(img, dtype="float32")
            og_img = numpy.array(og_img, dtype="float32")

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
            "og_img": og_img,
            "augmentations": self.augmentations[aug_idx],
            "identifier": f"{identifier}-{self.augmentations[aug_idx]}",
            "target_obj_file_path": trg_file,
        }

        if self.cache_images_in_memory:
            self.mem_cache[index] = item

        if self.verbose:
            deltaT(t1, "GI")

        return item
