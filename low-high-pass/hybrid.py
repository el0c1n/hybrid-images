"""
@author Nicole Schmelzer
2015
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import sys


def gauss_2d(shape=(3, 3), sigma=0.5):
    """
    generates gaussian filter matrix
    """
    filter_matrix = np.zeros(shape)
    nx, ny = filter_matrix.shape
    center_x = nx/2.0
    center_y = ny/2.0
    first_part = 2.0 * np.pi * np.power(sigma, 2.0)
    for x in range(shape[0]):
        for y in range(shape[1]):
            second_part = np.exp(- ((center_x - x)**2 + (center_y - y)**2) / (2. * sigma**2))
            filter_matrix[x, y] = second_part / first_part
    return filter_matrix


def generate_filter_matrix(no_rows, no_columns, sigma):
    """
    normalizes gaussian filter matrix (sum of entries = 1)
    """
    gauss_matrix = gauss_2d((no_rows, no_columns), sigma)
    return gauss_matrix/np.amax(gauss_matrix)


def generate_high_frequency_matrix(img, filter_matrix):
    """
    fourier transform image and shift lowest frequencies to center (where gaussian is 1),
    subtract from 1 to only let high frequencies through
    """
    return fftshift(fft2(img)) * (1 - filter_matrix)


def generate_low_frequency_matrix(img, filter_matrix):
    """
    fourier transform image and shift lowest frequencies to center (where gaussian is 1)
    """
    return fftshift(fft2(img)) * filter_matrix


def generate_hybrid_frequency_matrix(low_frequency_matrix, high_frequency_matrix):
    """
    merge low and high frequency matrices
    """
    hybrid_frequency_matrix = low_frequency_matrix + high_frequency_matrix
    return hybrid_frequency_matrix


def reverse_transform(frequency_image):
    """
    shift back and reverse fourier transform to get an image back
    """
    return ifft2(ifftshift(frequency_image)).real


def generate_color_images(low_pass_channels, high_pass_channels, hybrid_channels):
    """
    use reverse transform to get an image back for hybrid, low-pass and high-pass channels and merge to one image
    """
    low_pass_image = cv2.merge((reverse_transform(low_pass_channels[0]),
                               reverse_transform(low_pass_channels[1]),
                               reverse_transform(low_pass_channels[2])))
    high_pass_image = cv2.merge((reverse_transform(high_pass_channels[0]),
                                reverse_transform(high_pass_channels[1]),
                                reverse_transform(high_pass_channels[2])))
    hybrid_image = cv2.merge((reverse_transform(hybrid_channels[0]), reverse_transform(hybrid_channels[1]), reverse_transform(hybrid_channels[2])))
    return low_pass_image, high_pass_image, hybrid_image


def generate_greyscale_images(low_pass_channels, high_pass_channels, hybrid_channels):
    """
    use reverse transform to get an image back for hybrid, low-pass and high-pass
    """
    low_pass_image = reverse_transform(low_pass_channels[0])
    high_pass_image = reverse_transform(high_pass_channels[0])
    hybrid_image = reverse_transform(hybrid_channels[0])
    return low_pass_image, high_pass_image, hybrid_image


def generate(img1, img2, sigma1, sigma2):
    """
    generates low-pass, high-pass and hybrid image
    """
    filter_matrix_low = generate_filter_matrix(img1.shape[0], img1.shape[1], sigma1)
    filter_matrix_high = generate_filter_matrix(img1.shape[0], img1.shape[1], sigma2)

    # - split images in channels and generate low-pass/high-pass images for each of them
    low_frequency_channels = []
    high_frequency_channels = []
    for channel in cv2.split(img1):
        low_frequency_channels.append(generate_low_frequency_matrix(channel, filter_matrix_low))
    for channel in cv2.split(img2):
        high_frequency_channels.append(generate_high_frequency_matrix(channel, filter_matrix_high))

    # - generate hybrid image for each of the channels
    hybrid_channels = []
    for i in range(len(low_frequency_channels)):
        hybrid_channels.append(generate_hybrid_frequency_matrix(low_frequency_channels[i], high_frequency_channels[i]))

    # - generate greyscale or color images based on no of channels
    if len(hybrid_channels) == 1:
        return generate_greyscale_images(low_frequency_channels, high_frequency_channels, hybrid_channels)
    return generate_color_images(low_frequency_channels, high_frequency_channels, hybrid_channels)


SIGMA = 10

if __name__ == "__main__":
    sig1 = SIGMA
    sig2 = SIGMA

    if len(sys.argv) >= 2:
        if sys.argv[1] == 'c':
            source_img1 = cv2.imread('images/color_1.jpg')
            source_img2 = cv2.imread('images/color_2.jpg')
        else:
            source_img1 = cv2.imread('images/greyscale_1.png')
            source_img2 = cv2.imread('images/greyscale_2.png')
        if len(sys.argv) == 4:
            sig1 = float(sys.argv[2])
            sig2 = float(sys.argv[3])
    else:
        source_img1 = cv2.imread('images/greyscale_1.png')
        source_img2 = cv2.imread('images/greyscale_2.png')

    if source_img1.shape != source_img2.shape:
        print('Error: dimensions of images do not match: ', source_img1.shape, source_img2.shape)
    else:
        low_pass_image, high_pass_image, hybrid_image = generate(source_img1, source_img2, sig1, sig2)

        cv2.imwrite('low-pass.png', low_pass_image)
        cv2.imwrite('high-pass.png', high_pass_image)
        cv2.imwrite('hybrid.png', hybrid_image)
