import cv2
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import openslide
import os
import random
import skimage.color as sk_color
import skimage.filters as sk_filters
import skimage.io as sk_io
import skimage.morphology as sk_morphology
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import yaml

from albumentations import Transpose, RandomRotate90, ElasticTransform, GridDistortion, OpticalDistortion
from openslide import OpenSlideError
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage import measure
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
from torchvision import transforms
from torchvision import utils as vutils

current_path = os.path.abspath('.')
root_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(root_path)

BLACK = (0, 0, 0)
RED = (255, 0, 0)
LIME = (0, 255, 0)
BLUE = (0, 0 ,255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
SILVER = (192, 192, 192)
GRAY = (128, 128, 128)
MAROON = (128, 0, 0)
OLIVE = (128, 128, 0)
GREEN = (0, 128, 0)
PURPLE = (128, 0, 128)
TEAL = (0, 128, 128)
NAVY = (0, 0, 128)
CRIMSON = (220, 20, 60)
GOLDEN_ROD = (218, 165, 32)
SIENNA = (160, 82, 45)
PINK = (255, 192, 203)
GREEN_YELLOW = (173, 255, 47)
BEIGE = (245,245,220)
ORANGE = (255, 165, 0)
AZURE = (240, 255, 255)
DODGER_BLUE = (30, 144, 255)
VIOLET = (238, 130, 238)
CHOCOLATE = (210, 105, 30)
TOMATO = (255, 99, 71)
LIGHT_GREEN = (144, 238, 144)
DARK_SEA_GREEN = (143, 188, 143)
GOLD = (255, 215, 0)
WHITE = (255, 255, 255)
COLOR_CLASSES = np.array([BLACK, RED, LIME, BLUE, YELLOW, CYAN, MAGENTA, SILVER, GRAY, MAROON, OLIVE, GREEN, PURPLE,
                          TEAL, NAVY, CRIMSON, GOLDEN_ROD, SIENNA, PINK, GREEN_YELLOW, BEIGE, ORANGE, AZURE,
                          DODGER_BLUE, VIOLET, CHOCOLATE, TOMATO, LIGHT_GREEN, DARK_SEA_GREEN, GOLD, WHITE])


GREEN_COLOR = (0, 255, 0)
YELLOW_COLOR = (255, 255, 0)
ORANGE_COLOR = (255, 165, 0)
RED_COLOR = (255, 0, 0)


MAGNIFICATION_SCALE = {
    "20.0": 1.0,
    "10.0": 2.0,
    "5.0": 4.0,
    "2.5": 8.0,
    "1.25": 16.0,
    "0.625": 32.0,
    "0.3125": 64.0,
    "0.15625": 128.0,
    "0.078125": 256.0
}


###
# OPEN SLIDE FUNCTIONS
###
def get_scale_by_magnification(magnification):
    return MAGNIFICATION_SCALE[str(magnification)]


def open_wsi(filename):
    """
    Open a whole-slide image (*.svs, etc).
    Args:
        filename: Name of the image file.
    Returns:
        An OpenSlide object representing a whole-slide image.
    """

    try:
        slide = openslide.open_slide(filename)
    except OpenSlideError:
        slide = None
    except FileNotFoundError:
        slide = None

    return slide


def scale_down_wsi(wsi_image, magnification, use_openslide_propeties=True):
    """
    Convert a WSI to a scaled-down PIL image.
    Args:
        wsi_image: Whole-slide image to be scaled down.
        magnification: Whole-slide image magnification to be used.
        use_openslide_propeties:
    Returns:
        Returns the scaled-down PIL image.
    """
    scale = get_scale_by_magnification(magnification)
    if use_openslide_propeties:
        level = wsi_image.level_downsamples.index(scale)
        new_dimension = wsi_image.level_dimensions[level]
    else:
        large_w, large_h = wsi_image.dimensions
        new_w = math.floor(large_w / scale)
        new_h = math.floor(large_h / scale)
        new_dimension = (new_w, new_h)

    return wsi_image.get_thumbnail(new_dimension)


def scale_down_camelyon16_img(image_file, magnification):

    # load image
    wsi_image = open_wsi(image_file)

    # scales down the image
    scale = get_scale_by_magnification(magnification)
    wsi_image_pil = scale_down_wsi(wsi_image, magnification)

    return wsi_image_pil, scale


def extract_normal_region_from_wsi(wsi_image_file, np_scaled_down_image, np_tumor_mask):

    logger.info("\t Extracting normal regions from wsi image: '{}'".format(wsi_image_file.split('/')[-1]))

    np_mask = tissue_mask(np_scaled_down_image)
    if np_tumor_mask is not None:
        np_mask[np_tumor_mask > 0] = 0

    np_masked_image = mask_rgb(np_scaled_down_image, np_mask)

    return np_mask, np_masked_image


def read_region(wsi_image_file, column, row, magnification=0.625, tile_size=20):

    # load image
    wsi_image = open_wsi(wsi_image_file)
    max_w, max_h = wsi_image.dimensions

    scale = get_scale_by_magnification(magnification)
    level = wsi_image.get_best_level_for_downsample(scale)

    tile_size_original = int(tile_size*scale)
    left = (column * tile_size_original)
    top = (row * tile_size_original)
    tile_size_w = tile_size_original if (left + tile_size_original) <= max_w else (max_w - left)
    tile_size_h = tile_size_original if (top + tile_size_original) <= max_h else (max_h - top)

    region_pil = wsi_image.read_region((left, top), 0, (tile_size_w, tile_size_h))
    region_np = np.asarray(region_pil)

    if tile_size_w != tile_size_original or tile_size_h != tile_size_original:
        np_region = np.full((tile_size_original, tile_size_original, 3), 255, dtype=np.uint8)
        np_region[0:tile_size_h, 0:tile_size_w] = region_np[:, :, :3]

        return np_to_pil(np_region), np_region

    return region_pil, region_np[:, :, :3]


def draw_tile_border(draw, r_s, r_e, c_s, c_e, color=GREEN_COLOR, border_size=1, text=None):
    """
    Draw a border around a tile.
    Args:
        draw: Draw object for drawing on PIL image.
        r_s: Row starting pixel.
        r_e: Row ending pixel.
        c_s: Column starting pixel.
        c_e: Column ending pixel.
        color: RGB color of the border.
        border_size: Width of tile border in pixels.
        text: Label to draw into tile.
    """
    for x in range(0, border_size):
        draw.rectangle([(c_s + x, r_s + x), (c_e - 1 - x, r_e - 1 - x)], outline=color)

    #if text is not None:
    #    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 15)
    #    (x, y) = draw.textsize(text, font)
    #    draw.text((c_s + 5, r_s + 5), text, (255, 255, 255), font=font)


def draw_heat_grid(np_processed_img, tile_size):

    shape = np_processed_img.shape
    heat_grid = []
    tile_position = 0
    pil_processed_img = np_to_pil(np_processed_img)
    draw = ImageDraw.Draw(pil_processed_img)

    for height in range(0, shape[0], tile_size):
        for width in range(0, shape[1], tile_size):

            row = int(height / tile_size)
            column = int(width / tile_size)

            r_s = row * tile_size
            r_e = r_s + tile_size
            c_s = column * tile_size
            c_e = c_s + tile_size

            cropped_np_img = np_processed_img[r_s:r_e, c_s:c_e]
            tissue_area = tissue_percent(cropped_np_img)
            #            print("tile: {} - {}% r{} c{}".format(tile_position, tissue_area, row, column))

            if tissue_area <= 5.0:
                color = GREEN_COLOR
            elif 5.0 < tissue_area <= 10.0:
                color = YELLOW_COLOR
            elif 10.0 < tissue_area <= 80.0:
                color = ORANGE_COLOR
            else:
                color = RED_COLOR

            label = None
            if height == 0:
                label = str(int(width / tile_size))
            elif width == 0:
                label = str(int(height / tile_size))

            tile_position += 1
            location = (c_s, r_s)
            size = (tile_size, tile_size)
            tile = (tile_position, row, column, location, size, color)

            heat_grid.append(tile)
            draw_tile_border(draw, r_s, r_e, c_s, c_e, color, text=label)

    return pil_processed_img, heat_grid, tile_position
###


def extract_tumor_region_from_wsi(contours, wsi_image_file, magnification):

    logger.info("\t Extracting tumor regions from wsi image: '{}'".format(wsi_image_file.split('/')[-1]))

    wsi_image_pil, scale = scale_down_camelyon16_img(wsi_image_file, magnification)
    np_scaled_down_image = pil_to_np(wsi_image_pil)

    # find the tumor mask
    pil_mask = np_to_pil(np.zeros((np_scaled_down_image.shape[0], np_scaled_down_image.shape[1]), dtype=np.uint8))
    draw = ImageDraw.Draw(pil_mask)

    region_label = 1
    for idx, (region_name, annotation_type, group, color, points) in enumerate(contours):
        if group != "_2" and group != "Exclusion" and len(points) > 1:
            points_scaled_down = [tuple(pt * (1 / scale) for pt in p) for p in points]
            draw.polygon(points_scaled_down, outline=None, fill=region_label)
            region_label += 1
    for idx, (region_name, annotation_type, group, color, points) in enumerate(contours):
        if group == "_2" or group == "Exclusion" and len(points) > 1:
            points_scaled_down = [tuple(pt * (1 / scale) for pt in p) for p in points]
            draw.polygon(points_scaled_down, outline=None, fill=0)

    np_regions_label = pil_to_np(pil_mask).astype(np.uint8)
    np_mask = np_regions_label.astype(bool)
    #
    #np_mask = np_regions_label > 0
    np_regions_label = measure.label(np_mask, connectivity=2)
    #
    np_masked_image = mask_rgb(np_scaled_down_image, np_mask)

    return np_scaled_down_image, np_regions_label, np_mask, np_masked_image


def load_np_image(path, color_model="RGB"):

    pil_img = load_pil_image(path, gray=color_model == "GRAY", color_model=color_model)
    return pil_to_np(pil_img)


def load_pil_image(path, gray=False, color_model="RGB"):

    with open(path, 'rb') as f:

        if gray:
            return Image.open(f).convert('L')     # grayscale

        elif color_model == "HSV":
            # For HSV, 'H' range is [0, 179], 'S' range is [0, 255] and 'V' range is [0, 255]
            return Image.open(f).convert('HSV')      # hsv

        elif color_model == "LAB":
            rgb = sk_io.imread(path)
            if rgb.shape[2] > 3:  # removes the alpha channel
                rgb = sk_color.rgba2rgb(rgb)

            lab = sk_color.rgb2lab(rgb)
            # For LAB, 'L' range is [0, 100], 'A' range is [-127, 127] and 'B' range is [-127, 127]
            lab_scaled = ((lab + [0, 128, 128]) / [100, 255, 255])*255
            return Image.fromarray(lab_scaled.astype(np.uint8))

        return Image.open(f).convert('RGB')    # rgb


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), int(height))

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (int(width), int(h * r))

    #print("(h,w): {} / dim: {}".format((h,w), dim))
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def pil_to_np(pil_img):
    """
    Convert a PIL Image to a NumPy array.
    Note that RGB PIL (w, h) -> NumPy (h, w, 3).
    Args:
    pil_img: The PIL Image.
    Returns:
    The PIL image converted to a NumPy array.
    """

    rgb = np.asarray(pil_img)
    return rgb


def np_to_pil(np_img):
    """
    Convert a NumPy array to a PIL Image.
    Args:
        np_img: The image represented as a NumPy array.
    Returns:
    The NumPy array converted to a PIL Image.
    """

    if np_img.dtype == "bool":
        np_img = np_img.astype("uint8") * 255
    elif np_img.dtype == "float64":
        np_img = (np_img * 255).astype("uint8")

    return Image.fromarray(np_img)


def rgb_to_hsv(np_img):
    """
    Filter RGB channels to HSV (Hue, Saturation, Value).
    Args:
        np_img: RGB image as a NumPy array.
    Returns:
        Image as NumPy array in HSV representation.
    """

    return sk_color.rgb2hsv(np_img)


def rgb_to_lab(np_img):
    """
    Filter RGB channels to CIE L*a*b*.
    Args:
        np_img: RGB image as a NumPy array.
    Returns:
        Image as NumPy array in Lab representation.
    """

    if np_img.shape[2] > 3:  # removes the alpha channel
        np_img = sk_color.rgba2rgb(np_img)

    lab = sk_color.rgb2lab(np_img)
    # For LAB, 'L' range is [0, 100], 'A' range is [-127, 127] and 'B' range is [-127, 127]
    lab = ((lab + [0, 128, 128]) / [100, 255, 255])
    return lab


def lab_to_rgb(np_img):
    """
    Filter LAB channels to RGB (Red, Green, Blue).
    Args:
        np_img: LAB image as a NumPy array.
    Returns:
        Image as NumPy array in RGB representation.
    """

    lab_rescaled = ((np_img - [0, 128, 128]) * [100, 255, 255])/255
    rgb = sk_color.lab2rgb(lab_rescaled)
    return rgb


def hsv_to_rgb(np_img):
    """
    Filter HSV channels to RGB (Red, Green, Blue).
    Args:
        np_img: HSV image as a NumPy array.
    Returns:
        Image as NumPy array in RGB representation.
    """

    return sk_color.hsv2rgb(np_img)


def transfer_color(np_original_img_lab, np_target_img_lab, L_threshold=0.86):

    original_img_cbar_l = np_original_img_lab[:, :, 0][(np_original_img_lab[:, :, 0] < L_threshold)].mean()
    original_img_cbar_a = np_original_img_lab[:, :, 1][(np_original_img_lab[:, :, 0] < L_threshold)].mean()
    original_img_cbar_b = np_original_img_lab[:, :, 2][(np_original_img_lab[:, :, 0] < L_threshold)].mean()

    target_img_cbar_l = np_target_img_lab[:, :, 0][(np_target_img_lab[:, :, 0] < L_threshold)].mean()
    target_img_cbar_a = np_target_img_lab[:, :, 1][(np_target_img_lab[:, :, 0] < L_threshold)].mean()
    target_img_cbar_b = np_target_img_lab[:, :, 2][(np_target_img_lab[:, :, 0] < L_threshold)].mean()

    original_img_psc = np.copy(np_original_img_lab)
    original_img_psc[:, :, 0][(np_original_img_lab[:, :, 0] >= L_threshold)] = 0
    original_img_psc[:, :, 1][(np_original_img_lab[:, :, 0] >= L_threshold)] = -127
    original_img_psc[:, :, 2][(np_original_img_lab[:, :, 0] >= L_threshold)] = -127

    target_img_psc = np.copy(np_target_img_lab)
    target_img_psc[:, :, 0][(np_target_img_lab[:, :, 0] >= L_threshold)] = 0
    target_img_psc[:, :, 1][(np_target_img_lab[:, :, 0] >= L_threshold)] = -127
    target_img_psc[:, :, 2][(np_target_img_lab[:, :, 0] >= L_threshold)] = -127

    augmented_img = np.copy(np_original_img_lab)
    augmented_img[:, :, 0][(np_original_img_lab[:, :, 0] < L_threshold)] = augmented_img[:, :, 0][(np_original_img_lab[:, :, 0] < L_threshold)] - original_img_cbar_l + target_img_cbar_l
    augmented_img[:, :, 1][(np_original_img_lab[:, :, 0] < L_threshold)] = augmented_img[:, :, 1][(np_original_img_lab[:, :, 0] < L_threshold)] - original_img_cbar_a + target_img_cbar_a
    augmented_img[:, :, 2][(np_original_img_lab[:, :, 0] < L_threshold)] = augmented_img[:, :, 2][(np_original_img_lab[:, :, 0] < L_threshold)] - original_img_cbar_b + target_img_cbar_b

    return original_img_psc, target_img_psc, augmented_img



def filter_purple_pink(np_img, output_type="bool"):
    """
    Create a mask to filter out pixels where the values are similar to purple and pink.
    Args:
        np_img: RGB image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array representing a mask where pixels with purple/pink values have been masked out.
    """

    np_img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(np_img_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (130, 30, 50), (170, 255, 255))
    mask = basic_threshold(mask, threshold=0, output_type="bool")

    return parse_output_type(mask, output_type)


def remove_small_objects(np_img, min_size=3000, output_type="bool"):
    """
    Filter image to remove small objects (connected components) less than a particular minimum size.
    Args:
        np_img: Image as a NumPy array of type bool.
        min_size: Minimum size of small object to remove.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array (bool, float, or uint8).
    """

    result = np_img.astype(bool)  # make sure mask is boolean
    result = sk_morphology.remove_small_objects(result, min_size=min_size)
    return parse_output_type(result, output_type)


def fill_small_holes(np_img, area_threshold=3000, output_type="bool"):
    """
    Filter image to remove small holes less than a particular size.
    Args:
        np_img: Image as a NumPy array of type bool.
        area_threshold: Remove small holes below this area.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array (bool, float, or uint8).
    """

    result = sk_morphology.remove_small_holes(np_img, area_threshold=area_threshold)
    return parse_output_type(result, output_type)


def tissue_mask(np_img):

    # To prevent selecting background patches, slides are converted to HSV, blurred,
    # and patches filtered out if maximum pixel saturation lies below 0.07
    # (which was validated to not throw out tumor data in the training set).

    np_tissue_mask = filter_purple_pink(np_img)
    np_tissue_mask = fill_small_holes(np_tissue_mask, area_threshold=3000 if np_img.shape[0] > 500 else 30)
    np_tissue_mask = remove_small_objects(np_tissue_mask, min_size=3000 if np_img.shape[0] > 500 else 30)
    return np_tissue_mask


def mask_rgb(rgb, mask):
    """
    Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.
    Args:
        rgb: RGB image as a NumPy array.
        mask: An image mask to determine which pixels in the original image should be displayed.
    Returns:
        NumPy array representing an RGB image with mask applied.
    """

    result = rgb * np.dstack([mask, mask, mask])
    return result


def blend_image(image, mask, foreground='red', alpha=0.3, inverse=False):

    if inverse:
        mask = ImageOps.invert(mask)

    foreground = Image.new('RGB', image.size, color=foreground)
    composite = Image.composite(image, foreground, mask)
    return Image.blend(image, composite, alpha)


def mask_percent(np_img):
    """
    Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).
    Args:
        np_img: Image as a NumPy array.
    Returns:
        The percentage of the NumPy array that is masked.
    """

    if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
        np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
        mask_percentage = 0 if np_sum.size == 0 else 100 - np.count_nonzero(np_sum) / np_sum.size * 100
    else:
        mask_percentage = 0 if np_img.size == 0 else 100 - np.count_nonzero(np_img) / np_img.size * 100

    return mask_percentage


def tissue_percent(np_img):
    """
    Determine the percentage of a NumPy array that is tissue (not masked).
    Args:
        np_img: Image as a NumPy array.
    Returns:
        The percentage of the NumPy array that is tissue.
    """

    return 100 - mask_percent(np_img)


def basic_threshold(np_img, threshold=0.0, output_type="bool"):
    """
    Return mask where a pixel has a value if it exceeds the threshold value.
    Args:
        np_img: Binary image as a NumPy array.
        threshold: The threshold value to exceed.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array representing a mask where a pixel has a value (T, 1.0, or 255) if the corresponding input array pixel exceeds the threshold value.
    """

    result = (np_img > threshold)
    return parse_output_type(result, output_type)


def otsu_threshold(np_img, output_type="bool"):
    """
    Compute Otsu threshold on image as a NumPy array and return binary image based on pixels above threshold.
    Args:
        np_img: Image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above Otsu threshold.
    """

    otsu_thresh_value = sk_filters.threshold_otsu(np_img)
    result = (np_img > otsu_thresh_value)
    return parse_output_type(result, output_type)


def parse_output_type(np_img, output_type="bool"):
    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    return np_img


def output_map_to_rgb_image(output_map):

    r = np.zeros_like(output_map).astype(np.uint8)
    g = np.zeros_like(output_map).astype(np.uint8)
    b = np.zeros_like(output_map).astype(np.uint8)

    colors = np.copy(COLOR_CLASSES)
    qtd = len(np.unique(output_map)) - len(colors)
    if qtd > 0:
        colors = np.append(colors, COLOR_CLASSES[1:qtd], axis=0)
    else:
        colors = colors[0:len(COLOR_CLASSES)]

    for cls in range(0, len(colors)):
        idx = output_map == cls
        r[idx] = colors[cls, 0]
        g[idx] = colors[cls, 1]
        b[idx] = colors[cls, 2]
        rgb = np.stack([r, g, b], axis=2)

    return rgb


def show_np_img(np_img, text=None):
    """
    Convert a NumPy array to a PIL image, add text to the image, and display the image.
    Args:
        np_img: Image as a NumPy array.
        text: The text to be added to the image.
    """

    pil_img = np_to_pil(np_img)
    show_pil_img(pil_img, text)


def show_pil_img(pil_img, text=None):
    """
    Add text to the image, and display the image.
    Args:
        pil_img: PIL Image.
        text: The text to be added to the image.
    """

    # if gray, convert to RGB for display
    if pil_img.mode == 'L':
        pil_img = pil_img.convert('RGB')

    if text is not None:
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 48)
        (x, y) = draw.textsize(text, font)
        draw.rectangle([(0, 0), (x + 5, y + 4)], fill=(0, 0, 0), outline=(0, 0, 0))
        draw.text((2, 0), text, (255, 0, 0), font=font)

    pil_img.show()

formatter = logging.Formatter('%(asctime)s :: %(levelname)s %(funcName)s :: %(message)s')

# file handler
fh = logging.FileHandler('application.log')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)

# console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(ch)
logger.addHandler(fh)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def load_pil_image(path, gray=False, color_model="RGB"):

    with open(path, 'rb') as f:

        if gray:
            return Image.open(f).convert('L')     # grayscale

        elif color_model == "HSV":
            # For HSV, 'H' range is [0, 179], 'S' range is [0, 255] and 'V' range is [0, 255]
            return Image.open(f).convert('HSV')      # hsv

        elif color_model == "LAB":
            rgb = sk_io.imread(path)
            if rgb.shape[2] > 3:  # removes the alpha channel
                rgb = sk_color.rgba2rgb(rgb)

            lab = sk_color.rgb2lab(rgb)
            # For LAB, 'L' range is [0, 100], 'A' range is [-127, 127] and 'B' range is [-127, 127]
            lab_scaled = ((lab + [0, 128, 128]) / [100, 255, 255])*255
            return Image.fromarray(lab_scaled.astype(np.uint8))

        return Image.open(f).convert('RGB')    # rgb


def default_loader(path):
    return pil_loader(path)


def tensor_img_to_npimg(tensor_img):
    """
    Turn a tensor image with shape CxHxW to a numpy array image with shape HxWxC
    :param tensor_img:
    :return: a numpy array image with shape HxWxC
    """
    if not (torch.is_tensor(tensor_img) and tensor_img.ndimension() == 3):
        raise NotImplementedError("Not supported tensor image. Only tensors with dimension CxHxW are supported.")
    npimg = np.transpose(tensor_img.numpy(), (1, 2, 0))
    npimg = npimg.squeeze()
    assert isinstance(npimg, np.ndarray) and (npimg.ndim in {2, 3})
    return npimg


# Change the values of tensor x from range [0, 1] to [-1, 1]
def normalize(x):
    return x.mul_(2).add_(-1)

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def random_bbox(config, batch_size):
    """Generate a random tlhw with configuration.

    Args:
        config: Config should have configuration including img

    Returns:
        tuple: (top, left, height, width)

    """
    img_height, img_width, _ = config['image_shape']
    h, w = config['mask_shape']
    margin_height, margin_width = config['margin']
    maxt = img_height - margin_height - h
    maxl = img_width - margin_width - w
    bbox_list = []
    if config['mask_batch_same']:
        t = np.random.randint(margin_height, maxt)
        l = np.random.randint(margin_width, maxl)
        bbox_list.append((t, l, h, w))
        bbox_list = bbox_list * batch_size
    else:
        for i in range(batch_size):
            t = np.random.randint(margin_height, maxt)
            l = np.random.randint(margin_width, maxl)
            bbox_list.append((t, l, h, w))

    return torch.tensor(bbox_list, dtype=torch.int64)


def test_random_bbox():
    image_shape = [250, 450, 3]
    mask_shape = [128, 128]
    margin = [0, 0]
    config = {'image_shape': image_shape,
        'mask_shape': mask_shape,
        'margin': margin,
        'mask_batch_same': True
    }
    batch_size = 1  # ou outro valor adequado
    bbox = random_bbox(config, batch_size)
    return bbox


def bbox2mask(bboxes, height, width, max_delta_h, max_delta_w):
    batch_size = bboxes.size(0)
    mask = torch.zeros((batch_size, 1, height, width), dtype=torch.float32)
    for i in range(batch_size):
        bbox = bboxes[i]
        delta_h = np.random.randint(max_delta_h // 2 + 1)
        delta_w = np.random.randint(max_delta_w // 2 + 1)
        mask[i, :, bbox[0] + delta_h:bbox[0] + bbox[2] - delta_h, bbox[1] + delta_w:bbox[1] + bbox[3] - delta_w] = 1.
    return mask


def test_bbox2mask():
    image_shape = [250, 450, 3]
    mask_shape = [128, 128]
    margin = [0, 0]
    max_delta_shape = [32, 32]
    config = {
        'image_shape': image_shape,
        'mask_shape': mask_shape,
        'margin': margin,
        'mask_batch_same': True
    }
    batch_size = 1  # ou outro valor adequado
    bbox = random_bbox(config, batch_size)
    mask = bbox2mask(bbox, image_shape[0], image_shape[1], max_delta_shape[0], max_delta_shape[1])
    return mask


def local_patch(x, bbox_list):
    assert len(x.size()) == 4
    patches = []
    for i, bbox in enumerate(bbox_list):
        t, l, h, w = bbox
        patches.append(x[i, :, t:t + h, l:l + w])
    return torch.stack(patches, dim=0)


def mask_image(x, bboxes, config):
    height, width, _ = config['image_shape']
    max_delta_h, max_delta_w = config['max_delta_shape']
    mask = bbox2mask(bboxes, height, width, max_delta_h, max_delta_w)
    if x.is_cuda:
        mask = mask.cuda()

    if config['mask_type'] == 'hole':
        result = x * (1. - mask)
    elif config['mask_type'] == 'mosaic':
        # TODO: Matching the mosaic patch size and the mask size
        mosaic_unit_size = config['mosaic_unit_size']
        downsampled_image = F.interpolate(x, scale_factor=1. / mosaic_unit_size, mode='nearest')
        upsampled_image = F.interpolate(downsampled_image, size=(height, width), mode='nearest')
        result = upsampled_image * mask + x * (1. - mask)
    else:
        raise NotImplementedError('Not implemented mask type.')

    return result, mask


def spatial_discounting_mask(config):
    """Generate spatial discounting mask constant.

    Spatial discounting mask is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        config: Config should have configuration including HEIGHT, WIDTH,
            DISCOUNTED_MASK.

    Returns:
        tf.Tensor: spatial discounting mask

    """
    gamma = config['spatial_discounting_gamma']
    height, width = config['mask_shape']
    shape = [1, 1, height, width]
    if config['discounted_mask']:
        mask_values = np.ones((height, width))
        for i in range(height):
            for j in range(width):
                mask_values[i, j] = max(
                    gamma ** min(i, height - i),
                    gamma ** min(j, width - j))
        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 0)
    else:
        mask_values = np.ones(shape)
    spatial_discounting_mask_tensor = torch.tensor(mask_values, dtype=torch.float32)
    if config['cuda']:
        spatial_discounting_mask_tensor = spatial_discounting_mask_tensor.cuda()
    return spatial_discounting_mask_tensor


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def pt_flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = torch.tensor(-999)
    maxv = torch.tensor(-999)
    minu = torch.tensor(999)
    minv = torch.tensor(999)
    maxrad = torch.tensor(-1)
    if torch.cuda.is_available():
        maxu = maxu.cuda()
        maxv = maxv.cuda()
        minu = minu.cuda()
        minv = minv.cuda()
        maxrad = maxrad.cuda()
    for i in range(flow.shape[0]):
        u = flow[i, 0, :, :]
        v = flow[i, 1, :, :]
        idxunknow = (torch.abs(u) > 1e7) + (torch.abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = torch.max(maxu, torch.max(u))
        minu = torch.min(minu, torch.min(u))
        maxv = torch.max(maxv, torch.max(v))
        minv = torch.min(minv, torch.min(v))
        rad = torch.sqrt((u ** 2 + v ** 2).float()).to(torch.int64)
        maxrad = torch.max(maxrad, torch.max(rad))
        u = u / (maxrad + torch.finfo(torch.float32).eps)
        v = v / (maxrad + torch.finfo(torch.float32).eps)
        # TODO: change the following to pytorch
        img = pt_compute_color(u, v)
        out.append(img)

    return torch.stack(out, dim=0)


def highlight_flow(flow):
    """Convert flow into middlebury color code image.
    """
    out = []
    s = flow.shape
    for i in range(flow.shape[0]):
        img = np.ones((s[1], s[2], 3)) * 144.
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        for h in range(s[1]):
            for w in range(s[1]):
                ui = u[h, w]
                vi = v[h, w]
                img[ui, vi, :] = 255.
        out.append(img)
    return np.float32(np.uint8(out))


def pt_highlight_flow(flow):
    """Convert flow into middlebury color code image.
        """
    out = []
    s = flow.shape
    for i in range(flow.shape[0]):
        img = np.ones((s[1], s[2], 3)) * 144.
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        for h in range(s[1]):
            for w in range(s[1]):
                ui = u[h, w]
                vi = v[h, w]
                img[ui, vi, :] = 255.
        out.append(img)
    return np.float32(np.uint8(out))


def compute_color(u, v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
    return img


def pt_compute_color(u, v):
    h, w = u.shape
    img = torch.zeros([3, h, w])
    if torch.cuda.is_available():
        img = img.cuda()
    nanIdx = (torch.isnan(u) + torch.isnan(v)) != 0
    u[nanIdx] = 0.
    v[nanIdx] = 0.
    # colorwheel = COLORWHEEL
    colorwheel = pt_make_color_wheel()
    if torch.cuda.is_available():
        colorwheel = colorwheel.cuda()
    ncols = colorwheel.size()[0]
    rad = torch.sqrt((u ** 2 + v ** 2).to(torch.float32))
    a = torch.atan2(-v.to(torch.float32), -u.to(torch.float32)) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = torch.floor(fk).to(torch.int64)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0.to(torch.float32)
    for i in range(colorwheel.size()[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1]
        col1 = tmp[k1 - 1]
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1. / 255.
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = (idx != 0)
        col[notidx] *= 0.75
        img[i, :, :] = col * (1 - nanIdx).to(torch.float32)
    return img


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def pt_make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = torch.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 1.
    colorwheel[0:RY, 1] = torch.arange(0, RY, dtype=torch.float32) / RY
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 1. - (torch.arange(0, YG, dtype=torch.float32) / YG)
    colorwheel[col:col + YG, 1] = 1.
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 1.
    colorwheel[col:col + GC, 2] = torch.arange(0, GC, dtype=torch.float32) / GC
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 1. - (torch.arange(0, CB, dtype=torch.float32) / CB)
    colorwheel[col:col + CB, 2] = 1.
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 1.
    colorwheel[col:col + BM, 0] = torch.arange(0, BM, dtype=torch.float32) / BM
    col += BM
    # MR
    colorwheel[col:col + MR, 2] = 1. - (torch.arange(0, MR, dtype=torch.float32) / MR)
    colorwheel[col:col + MR, 0] = 1.
    return colorwheel


def is_image_file(filename):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def deprocess(img):
    img = img.add_(1).div_(2)
    return img


# get configs
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


# Get model list for resume
def get_model_list(dirname, key, iteration=0):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    if iteration == 0:
        last_model_name = gen_models[-1]
    else:
        for model_name in gen_models:
            if '{:0>8d}'.format(iteration) in model_name:
                return model_name
        raise ValueError('Not found models with this iteration')
    return last_model_name


if __name__ == '__main__':
    test_random_bbox()
    mask = test_bbox2mask()
    print(mask.shape)

    mask_squeezed = mask.squeeze()

    plt.imshow(mask_squeezed, cmap='gray')
    plt.show()

def data_augmentation(input_image, target_img, output_mask, img_input_size=(250, 450), img_output_size=(250, 450), aug=None, GAN_model=None):

    image = TF.resize(input_image, size=img_output_size)
    target_image = TF.resize(target_img, size=img_output_size) if target_img is not None else None
    mask = TF.resize(output_mask, size=img_output_size) if output_mask is not None and np.any(np.unique(pil_to_np(output_mask) > 0)) else None

    used_augmentations = []
    if aug is not None and len(aug) > 0:

        # Random horizontal flipping
        if "horizontal_flip" in aug and (len(aug) < 2 or random.random() > 0.5):
            image = TF.hflip(image)
            mask = TF.hflip(mask) if mask is not None else None
            used_augmentations.append("horizontal_flip")

        # Random vertical flipping
        if "vertical_flip" in aug and (len(aug) < 2 or random.random() > 0.5):
            image = TF.vflip(image)
            mask = TF.vflip(mask) if mask is not None else None
            used_augmentations.append("vertical_flip")

        # Random rotation
        if "rotation" in aug and (len(aug) < 2 or random.random() > 0.5) and img_input_size[0] == img_input_size[1]:
            augmented = RandomRotate90(p=1)(image=np.array(image), mask=np.array(mask) if mask is not None else np.zeros(img_output_size))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])
            used_augmentations.append("rotation")

        # Random transpose
        if "transpose" in aug and (len(aug) < 2 or random.random() > 0.5) and img_input_size[0] == img_input_size[1]:
            augmented = Transpose(p=1)(image=np.array(image), mask=np.array(mask) if mask is not None else np.zeros(img_output_size))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])
            used_augmentations.append("transpose")

        # Random elastic transformation
        if "elastic_transformation" in aug and (len(aug) < 2 or random.random() > 0.5):
            alpha = random.randint(100, 200)
            augmented = ElasticTransform(p=1, alpha=alpha, sigma=alpha * 0.05, alpha_affine=alpha * 0.03)(image=np.array(image), mask=np.array(mask) if mask is not None else np.zeros(img_output_size))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])
            used_augmentations.append("elastic_transformation")

        # Random GridDistortion
        if "grid_distortion" in aug and (len(aug) < 2 or random.random() > 0.5):
            augmented = GridDistortion(p=1)(image=np.array(image), mask=np.array(mask) if mask is not None else np.zeros(img_output_size))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])
            used_augmentations.append("grid_distortion")

        # Random OpticalDistortion
        if "optical_distortion" in aug and (len(aug) < 2 or random.random() > 0.5):
            augmented = OpticalDistortion(p=1, distort_limit=1, shift_limit=0.5)(image=np.array(image),mask=np.array(mask) if mask is not None else np.zeros(img_output_size))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])
            used_augmentations.append("optical_distortion")

        # Color transfer augmentation
        if "color_transfer" in aug and target_image is not None and (len(aug) < 2 or random.random() > 0.5):

            original_img_lab = TF.to_tensor(image).permute(1, 2, 0).numpy()
            target_img_lab = TF.to_tensor(target_image).permute(1, 2, 0).numpy()

            _, _, augmented_img = transfer_color(original_img_lab, target_img_lab)
            image = transforms.ToPILImage()(torch.from_numpy(augmented_img).permute(2, 0, 1))
            used_augmentations.append("color_transfer")

        # Inpainting augmentation
        if "inpainting" in aug and (len(aug) < 2 or random.random() > 0.5):

            width, height = image.size
            sourcecode_dir = os.path.dirname(os.path.abspath('.'))
            config_file = os.path.join(sourcecode_dir, 'GAN/configs/config_imagenet_ocdc.yaml')
            config = get_config(config_file)

            # Setting the points for cropped image
            crop_size = config['image_shape']
            left = np.random.randint(0, width-crop_size[0])
            top = np.random.randint(0, height-crop_size[1])

            cropped_region = image.crop((left, top, left+crop_size[0], top+crop_size[1]))
            cropped_region = pil_to_np(cropped_region)
            cropped_region = lab_to_rgb(cropped_region)
            cropped_region = transforms.ToTensor()(cropped_region)
            inpainting_img = cropped_region.detach().clone().mul_(2).add_(-1)        # normalize between -1 and 1
            inpainting_img = inpainting_img.unsqueeze(dim=0).to(dtype=torch.float32) # adds the batch channel

            bboxes = random_bbox(config, batch_size=inpainting_img.size(0))
            inpainting_img, inpainting_mask = mask_image(inpainting_img, bboxes, config)

            if torch.cuda.is_available():
                GAN_model = nn.parallel.DataParallel(GAN_model)
                inpainting_img = inpainting_img.cuda()
                inpainting_mask = inpainting_mask.cuda()

            # Inpainting inference
            x1, x2, offset_flow = GAN_model(inpainting_img, inpainting_mask)
            inpainted_result = x2 * inpainting_mask + inpainting_img * (1. - inpainting_mask)
            inpainted_result = inpainted_result.squeeze(0).add_(1).div_(2) # renormalize between 0 and 1
            inpainted_result = transforms.ToTensor()(rgb_to_lab(inpainted_result.permute(1, 2, 0).cpu().detach().numpy()))

            #viz_images = torch.stack([inpainting_img, inpainted_result.unsqueeze(dim=0).cuda()], dim=1)
            #viz_images = viz_images.view(-1, *list(inpainting_img.size())[1:])
            #vutils.save_image(viz_images,
            #                    '/home/dalifreire/Pictures/augmentation/teste_%03d.png' % (random.randint(0, 999)),
            #                    nrow=2 * 4,
            #                    normalize=True)

            augmented_img = TF.to_tensor(image)
            augmented_img[:, top:top+crop_size[1], left:left+crop_size[0]] = inpainted_result.squeeze(0)
            image = transforms.ToPILImage()(augmented_img)
            used_augmentations.append("inpainting")

            #augmented_img[:, top:top+crop_size[1], left:left+crop_size[0]] = inpainting_img.squeeze(0)
            #transforms.ToPILImage()(augmented_img).save('/home/dalifreire/Pictures/augmentation/1009010x1000902_r39c49_augmented_inpainting_{}.png'.format((random.randint(0, 999))))


    # Transform to grayscale (1 channel)
    mask = TF.to_grayscale(mask, num_output_channels=1) if mask is not None else None

    # Transform to pytorch tensor and binarize the mask
    image = TF.to_tensor(image).float()

    unique_mask_values = np.unique(pil_to_np(mask))
    mask = torch.zeros(img_output_size) if mask is None or not np.any(unique_mask_values) else (torch.ones(img_output_size) if np.any(unique_mask_values) and unique_mask_values.size == 1 else TF.to_tensor(np_to_pil(basic_threshold(np_img=pil_to_np(mask)))).squeeze(0).float())

    return image, mask, used_augmentations

