"""
@author Nicole Schmelzer
2015
"""

import numpy as np
import cv2
import math


# image sizes = [nxn, n/2xn/2, ...]
def generate_gaussian_pyramid(img, level, pyramid):
    pyramid.append(img)
    m, n = img.shape[:-1] if len(img.shape) == 3 else img.shape

    if level == 0:
        return pyramid

    m_half = m//2 if m % 2 == 0 else m//2+1
    n_half = n//2 if n % 2 == 0 else n//2+1
    expanded_img = np.zeros((m_half, n_half, 3)) if len(img.shape) == 3 else np.zeros((m_half, n_half))
    blur = cv2.GaussianBlur(img, (13, 13), 0)
    i, j = 0,0
    for x in range(0, m, 2):
        for y in range(0, n, 2):
            expanded_img[i, j] = blur[x, y]
            j += 1
        i += 1
        j = 0
    return generate_gaussian_pyramid(expanded_img, level-1, pyramid)


# image sizes = [nxn, n/2xn/2, ...]
def generate_laplacian_pyramid(gaussian_pyramid):
    lap_pyr = []
    for i in range(len(gaussian_pyramid)):
        if i == len(gaussian_pyramid)-1:
            lap_pyr.append(gaussian_pyramid[i])
        else:
            img = gaussian_pyramid[i]
            m, n = img.shape[:-1] if len(img.shape) == 3 else img.shape

            expanded_img = cv2.resize(gaussian_pyramid[i+1], (n, m))
            lap_pyr.append(img - expanded_img)
    return lap_pyr


def calc_laplacian_pyramid(img, depth):
    return generate_laplacian_pyramid(generate_gaussian_pyramid(img, depth, []))


def reconstruct_from_laplace(pyramid):
    gauss_img = None
    pyramid = pyramid[::-1]

    for i in range(1, len(pyramid)):
        if i == 1:
            gauss_img = pyramid[i-1]
        m, n = pyramid[i].shape[:-1] if len(pyramid[i].shape) == 3 else pyramid[i].shape
        gauss_img = cv2.resize(gauss_img, (n, m))
        gauss_img = pyramid[i] + gauss_img

    return gauss_img


def join_laplacian_images(lp_img1, lp_img2):
    lp_pyr_joined = []
    for i in range(len(lp_img1)):
        lp_pyr_joined.append(calc_direct_blending(lp_img1[i], lp_img2[i]))
    return lp_pyr_joined


# - Blend the two laplacian pyramids by weighing them according to the mask
def blend(lapl_pyr_black, lapl_pyr_white, gauss_pyr_mask):
    blended_pyr = []
    for i in range(0, len(gauss_pyr_mask)):
        p1 = gauss_pyr_mask[i] * lapl_pyr_white[i]
        p2 = (1 - gauss_pyr_mask[i]) * lapl_pyr_black[i]
        blended_pyr.append(p1 + p2)
    return blended_pyr


def calc_direct_blending(img_left, img_right):
    m, n = img_left.shape[:-1] if len(img_left.shape) == 3 else img_left.shape

    half = n//2  # use // to generate an int not a float
    img_l_half = img_left[:, 0:half]
    img_r_half = img_right[:, half:]

    return np.append(img_l_half, img_r_half, axis=1)


def calc_laplacian_blending(img_left, img_right, use_mask=True):
    # Automatically figure out the size
    min_size = min(img_left.shape[:-1])
    depth = int(math.floor(math.log(min_size, 2))) - 4  # at least 16x16 at the highest level.

    lap_pyr_left = calc_laplacian_pyramid(img_left, depth)
    lap_pyr_right = calc_laplacian_pyramid(img_right, depth)

    if use_mask:
        mask = np.zeros(img_left.shape)
        mask[:, img_left.shape[1]//2:] = 1
        gaussian_mask = generate_gaussian_pyramid(mask, depth, [])

        blended = blend(lap_pyr_left, lap_pyr_right, gaussian_mask)
        reconstructed = reconstruct_from_laplace(blended)

        # blending sometimes results in slightly out of bound numbers.
        reconstructed[reconstructed < 0] = 0
        reconstructed[reconstructed > 255] = 255
        reconstructed = reconstructed.astype(np.uint8)

    else:
        joint_pyramid = join_laplacian_images(lap_pyr_left, lap_pyr_right)
        reconstructed = reconstruct_from_laplace(joint_pyramid)

    return reconstructed


if __name__ == "__main__":
    img_left  = cv2.imread('images/apple.jpg')
    img_right = cv2.imread('images/orange.jpg')

    direct_blend = calc_direct_blending(img_left, img_right)

    lap_mask_blend = calc_laplacian_blending(img_left, img_right)
    lap_blend = calc_laplacian_blending(img_left, img_right, False)

    cv2.imwrite('direct_blend.png', direct_blend)
    cv2.imwrite('reconstructed-mask.png', lap_mask_blend)
    cv2.imwrite('reconstructed.png', lap_blend)
