# -*- coding: utf-8 -*-
"""
Resizes and crops but doesn't upscale images to match given size.
"""

import cv2



def apply(img, size):
    h, w, _ = img.shape

    # Resize
    if h > size and w > size:
        min_side = min(h, w)
        resize_factor = size / min_side
        img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)

    h, w, _ = img.shape

    # Skip if both sides are smaller than size
    if h <= size and w <= size: return img

    # Crop if one side is smaller than size
    elif h <= size:
        margin = (w - size) // 2
        img = img[:, margin: margin + size]
    elif w <= size:
        margin = (h - size) // 2
        img = img[margin: margin + size, :]

    return img
