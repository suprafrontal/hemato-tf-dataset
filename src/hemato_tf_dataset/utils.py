import time
from PIL import Image


def deltaT(t1: float, msg: str) -> float:
    delta = time.time() - t1
    t1 = time.time()
    print(f"\n\x1B[32m ∆ {msg} {delta:0.3f}sec \x1b[0m\n")
    return t1


def translate_wrap(image:Image, tx:int, ty:int) -> Image:
    width, height = image.size
    new_image = Image.new(image.mode, image.size)

    for x in range(width):
        for y in range(height):
            # Apply the translation with wrap-around
            new_x = (x + tx) % width
            new_y = (y + ty) % height
            # Copy the pixel to the new image
            new_image.putpixel((new_x, new_y), image.getpixel((x, y)))
    return new_image
