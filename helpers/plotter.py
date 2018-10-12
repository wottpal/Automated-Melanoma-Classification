# -*- coding: utf-8 -*-
"""Helper for plotting multiple images."""

import math
import matplotlib.pyplot as plt
import cv2


def plt_show():
    """Workaround for nasty macOS-bug: https://github.com/matplotlib/matplotlib/issues/9637#issuecomment-363682221"""
    try:
        plt.show()
    except UnicodeDecodeError:
        plt_show()


def plot_images(images, title = ""):
    n = len(images)
    rows = int(math.sqrt(n))
    cols = n // rows
    fig = plt.figure(figsize=(2.5 * cols, 2.5 * rows))
    for i in range(1, n+1):
        image = images[i-1]
        fig.add_subplot(rows, cols, i)
        plt.xticks([]),plt.yticks([])
        plt.imshow(image)
    fig.tight_layout()
    plt_show()


def plot_bgr_images(images, title = ""):
    conversionMode = lambda img: cv2.COLOR_GRAY2RGB if len(img.shape) == 2 else cv2.COLOR_BGR2RGB
    images = [cv2.cvtColor(img, conversionMode(img)) for img in images]
    plot_images(images, title)
