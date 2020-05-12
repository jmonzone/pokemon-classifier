from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2
import math

def lbp3x3(image, partitions=3):
    height, width, channel = image.shape
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_lbp = np.zeros((height, width,3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
             img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)

    hists = []
    image_slices = []
    for i in range(partitions**2):
        partition = getPartition(img_lbp, i, partitions)
        (hist, _) = np.histogram(partition, bins=254, range=(0, 254))
        hists.extend(hist)
        image_slices.append(partition)

    return (img_lbp, hists, image_slices)

def lbp9x9(image, partitions=3):
    height, width, channel = image.shape
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_lbp = np.zeros((height, width,3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp9x9_calculated_pixel(img_gray, i, j)

    hists = []
    image_slices = []
    for i in range(partitions**2):
        partition = getPartition(img_lbp, i, partitions)
        (hist, _) = np.histogram(partition, bins=254, range=(0, 254))
        hists.extend(hist)
        image_slices.append(partition)

    return (img_lbp, hists, image_slices)

def getBoxCounts(image, partition_dimensions = [16,24,32,40,48,56,64]):
    image = cv2.blur(image,(5,5))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(image,254,255,cv2.THRESH_BINARY)

    box_counts = []
    for k in range(len(partition_dimensions)):
        partiton_dimension = partition_dimensions[k]
        partition_count = partiton_dimension ** 2
        box_count = 0

        for i in range(partition_count):
            partition = getPartition(thresh, i, partiton_dimension)
            if 0 in partition:
                box_count += 1

        box_counts.append(box_count)

    return box_counts

def colorHist(image):
    hists = []
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [255], [1, 255])
        hists.append(hist)
    hists = np.asarray(hists)
    return hists

def colorMean(image):
    channels = cv2.mean(image)
    mean = [channels[0], channels[1], channels[2]]
    return mean

def dominantColor(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pixels = np.float32(image.reshape(-1, 3))
    background_color = [0,0,0]
    mask = np.all(pixels != background_color, axis = -1)
    pixels = pixels[mask]
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    return (dominant,palette,counts)


def gaussPyramid(image, levels=6):
    resos = [image]
    for i in range(levels):
        image = cv2.pyrDown(image)
        resos.append(image)
    return resos

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val
def get_pixel_value(img, x, y):
    value = -1
    try:
        value = img[x][y]
    except:
        pass
    return value

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val

def getPartition(img, index, partitions=9):
    h = img.shape[0]
    w = img.shape[1]
    index_x = index % partitions
    index_y = math.floor(index / partitions)
    xratio1 = index_x/partitions
    xratio2 = (index_x+1)/partitions
    yratio1 = index_y/partitions
    yratio2 = (index_y+1)/partitions
    part = img[int(yratio1*h):int(yratio2*h),int(xratio1*w):int(xratio2*w)].copy()
    return part

def getCenterBlockAverage(img, x, y):
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel_value(img, x-1, y+1))     # top_right
    val_ar.append(get_pixel_value(img, x, y+1))       # right
    val_ar.append(get_pixel_value(img, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel_value(img, x+1, y))       # bottom
    val_ar.append(get_pixel_value(img, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel_value(img, x, y-1))       # left
    val_ar.append(get_pixel_value(img, x-1, y-1))     # top_left
    val_ar.append(get_pixel_value(img, x-1, y))       # top

    val_ar = [item for item in val_ar if item >= 0]

    val = sum(val_ar) / len(val_ar)
    return val

def getBlockAverage(img, center, x, y):
    new_value = 0
    try:
        sub_center = img[x][y]
        val_ar = []
        val_ar.append(get_pixel_value(img, x-1, y+1))     # top_right
        val_ar.append(get_pixel_value(img, x, y+1))       # right
        val_ar.append(get_pixel_value(img, x+1, y+1))     # bottom_right
        val_ar.append(get_pixel_value(img, x+1, y))       # bottom
        val_ar.append(get_pixel_value(img, x+1, y-1))     # bottom_left
        val_ar.append(get_pixel_value(img, x, y-1))       # left
        val_ar.append(get_pixel_value(img, x-1, y-1))     # top_left
        val_ar.append(get_pixel_value(img, x-1, y))       # top

        val_ar = [item for item in val_ar if item >= 0]

        val = sum(val_ar) / len(val_ar)

        if val >= center:
            new_value = 1
    except:
        pass
    return new_value


def lbp9x9_calculated_pixel(img, x, y):
    center = getCenterBlockAverage(img, x, y)

    val_ar = []
    val_ar.append(getBlockAverage(img, center, x-3, y+3))     # top_right
    val_ar.append(getBlockAverage(img, center, x, y+3))       # right
    val_ar.append(getBlockAverage(img, center, x+3, y+3))     # bottom_right
    val_ar.append(getBlockAverage(img, center, x+3, y))       # bottom
    val_ar.append(getBlockAverage(img, center, x+3, y-3))     # bottom_left
    val_ar.append(getBlockAverage(img, center, x, y-3))       # left
    val_ar.append(getBlockAverage(img, center, x-3, y-3))     # top_left
    val_ar.append(getBlockAverage(img, center, x-3, y))       # top

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val
