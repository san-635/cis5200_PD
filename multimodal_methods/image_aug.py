# some code in this script is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random
import PIL, PIL.ImageOps, PIL.ImageEnhance

def TranslateXabs(img, v):
    """ Translate the image along the horizontal axis (small pixel shift). """
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateYabs(img, v):
    """ Translate the image along the vertical axis (small pixel shift). """
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def Flip(img, _):
    """ Flip the image horizontally (left-right). """
    return PIL.ImageOps.mirror(img)

def Brightness(img, v):
    """ Adjust the brightness of the image. """
    assert 0.7 <= v <= 1.3   # narrower range for medical images (optional)
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def augment_list():
    """ Augmentations allowed: small translations, horizontal flip, brightness. """
    ops = [
        (TranslateXabs, 0.0, 10),   # pixel shift limits; tune as needed
        (TranslateYabs, 0.0, 10),
        (Flip,         0,   1),
        (Brightness,   0.7, 1.3),
    ]
    return ops

class RandAug:
    """ Randomly apply a set of data augmentations to an image. """
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
        return img