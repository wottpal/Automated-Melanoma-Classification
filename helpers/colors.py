# -*- coding: utf-8 -*-
"""Helper Functions for generating random colors.

Source: https://gist.github.com/adewes/5884820
"""

import random
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw


def get_random_color(pastel_factor=0.9):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0, 1.0) for i in [1, 2, 3]]]


def color_distance(c1, c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1, c2)])


def generate_new_color(existing_colors):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color()
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c)
                             for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color


def get_random_colors(amount):
    colors = []
    random.seed(10)
    for _ in range(amount):
        colors.append(generate_new_color(colors))
    return colors


def generate_gradient_image(w, h):
    """Source: https://stackoverflow.com/a/31125282/1381666"""
    im = Image.new('RGB', (w, h))
    ld = im.load()

    colors = [
        [0.0, (0, 1., 0)],
        [0.5, (.75, .75, 0)],
        [1.0, (1., 0, 0)],
    ]

    def gaussian(x, a, b, c, d=0):
        return a * math.exp(-(x - b)**2 / (2 * c**2)) + d

    def pixel(x, width, map=[], spread=1):
        width = float(width)
        r = sum([gaussian(x, p[1][0], p[0] * width, width/(spread*len(map))) for p in map])
        g = sum([gaussian(x, p[1][1], p[0] * width, width/(spread*len(map))) for p in map])
        b = sum([gaussian(x, p[1][2], p[0] * width, width/(spread*len(map))) for p in map])
        return min(1.0, r), min(1.0, g), min(1.0, b)

    for x in range(im.size[0]):
        r, g, b = pixel(x, width=w, map=colors)
        r, g, b = [int(256*v) for v in (r, g, b)]
        for y in range(im.size[1]):
            ld[x, y] = r, g, b
    
    return im


# def applyTrafficLightColorMap(gray):
#     '''Reference: https://github.com/radjkarl/imgProcessor/blob/master/imgProcessor/transformations.py'''
#     mx = 256  # if gray.dtype==np.uint8 else 65535
#     lut = np.empty(shape=(256, 3))

#     cmap = (
#         (0, (0, 255, 0)),
#         (.5, (191, 191, 0)),
#         (1, (255, 0, 0))
#     )

#     lastval, lastcol = cmap[0]
#     for step, col in cmap[1:]:
#         val = int(step * mx)
#         for i in range(3):
#             lut[lastval:val, i] = np.linspace(lastcol[i], col[i], val - lastval)

#         lastcol = col
#         lastval = val

#     s0, s1 = gray.shape
#     out = np.empty(shape=(s0, s1, 3), dtype=np.uint8)

#     for i in range(3):
#         out[..., i] = cv2.LUT(gray, lut[:, i])
#     return out
