from PIL import Image
from PIL import *
import cv2
import numbers
import numpy as np
import torch


def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped image.
        w (int): Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    return img.crop((j, i, j + w, i + h))


def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)


def cv2_crop(img, output_size):
    h, w = img.shape[0], img.shape[1]
    th, tw = output_size, output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return img[i:i + tw, j:j + tw]


def resize(img, size, interpolation=Image.BILINEAR):
    r"""Resize the input PIL Image to the given size.
    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    Returns:
        PIL Image: Resized image.
    """
    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def normalize(PILImg, mean, std):
    '''
    PILImg,  PIL image

    '''
    img = np.array(PILImg)
    img = img.transpose(2, 0, 1) / 255.0
    img[0, :, :] = (img[0, :, :] - mean[0]) / std[0]
    img[1, :, :] = (img[1, :, :] - mean[1]) / std[1]
    img[2, :, :] = (img[2, :, :] - mean[2]) / std[2]
    return torch.from_numpy(img).float()


def cv2Norm(img, mean, std):
    img = img.transpose(2, 0, 1) / 255.0
    img[0, :, :] = (img[0, :, :] - mean[0]) / std[0]
    img[1, :, :] = (img[1, :, :] - mean[1]) / std[1]
    img[2, :, :] = (img[2, :, :] - mean[2]) / std[2]
    return torch.from_numpy(img).float()


def cv2Resize(img, size):
    h, w = img.shape[0], img.shape[1]
    #if (w <= h and w == size) or (h <= w and h == size):
    if max(w,h) <= size:
        return img
    if h < w:
        ow = size
        oh = int(size * h / w)
        return cv2.resize(img, (ow, oh))
    else:
        oh = size
        ow = int(size * w / h)
        return cv2.resize(img, (ow, oh))


def VQEResize(img, size):
    h, w = img.shape[0], img.shape[1]
    if max(w,h) <= size:
        return img

    if h < w:
        tw = size
        th = int(size * h / w)
    else:
        th = size
        tw = int(size * w / h)
         
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return img[i:i + th, j:j + tw]


def myTorchTransform(self, img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        cv2Img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
        mean_rgb = (0.485, 0.456, 0.406)
        std_rgb = (0.229, 0.224, 0.225)
        img = MYT.center_crop(MYT.resize(img, 265), 244)
        img = MYT.normalize(img, mean_rgb, std_rgb)
        return img, cv2Img, img_path
    except Exception as err:
        print('[ ERROR ] :: myTorchTransform error {}'.format(err))
        return None, None, None


def myCv2TorchTransform(self, img_path):
    try:
        t1 = time.time()
        img = cv2.imread(img_path)
        t1 = time.time()
        img = resizeImage(img)

        t1 = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mean_rgb = (0.485, 0.456, 0.406)
        std_rgb = (0.229, 0.224, 0.225)

        t1 = time.time()
        org_img = img

        img = MYT.cv2_crop(MYT.cv2Resize(org_img, 265), 244)
        img = MYT.cv2Norm(img, mean_rgb, std_rgb)

        del org_img

        return img, gray, img_path
    except Exception as err:
        print('[ ERROR ] :: myTorchTransform error {}'.format(err))
        return None, None, None


def myCv2TorchTransformV2(self, img, img_path):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mean_rgb = (0.485, 0.456, 0.406)
        std_rgb = (0.229, 0.224, 0.225)

        org_img = img

        img = MYT.cv2_crop(MYT.cv2Resize(org_img, 265), 244)
        img = MYT.cv2Norm(img, mean_rgb, std_rgb)

        del org_img

        return img, gray, img_path
    except Exception as err:
        print('[ ERROR ] :: myTorchTransform error {}'.format(err))
        return None, None, None
