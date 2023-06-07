"""
        '########:'##::::'##::'########:::
         ##.....::. ##::'##:::::.....##:::
         ##::::::::. ##'##:::::::::::##:::
         ######:::::. ###::::. ########:::
         ##...:::::: ## ##:::::......##::::
         ##:::::::: ##:. ##::::::::::##:::
         ########: ##:::. ##:. ########:::
        ........::..:::::..:::.........:::
"""



import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


GRADIENT = np.array([[1, 0, -1]])


def myID() -> np.int32:
    return 204155311

# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------

# Auxolarity functions
def __gradient(img: np.ndarray):
    I_x = cv2.filter2D(img, -1, GRADIENT)
    I_y = cv2.filter2D(img, -1, GRADIENT.T)
    return I_x, I_y
def is_window_valid(eigen_value: np.ndarray) -> bool:
    lamda1 = np.max(eigen_value)
    lamda2 = np.min(eigen_value)
    if lamda2 <= 1 or lamda1 / lamda2 >= 100:
        return False
    return True
def __eigenvalues_checking(At_mult_A: np.ndarray) -> bool:
    """
    1. Get the At_mult_A - a squred matrix.
    2. Find the eigen values.
        * Using np.linalg.eigvals, not np.linalg.eig, for getting just eigen values, without eigen vectors.
    3. Return validation checking result (with is_window_valid() func)
    """
    eigen_values = np.linalg.eigvals(At_mult_A)
    return is_window_valid(eigen_values)


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> tuple[np.ndarray, np.ndarray]:
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    # Generate a gray-scale copy
    img1, img2 = im1, im2
    if len(im1.shape) == 3:
        img1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    if len(im2.shape) == 3:
        img2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    
    # Optimize LK value by the Iterative Algorithm
    #   1. Calculate x, y gradiante
    I_x, I_y = __gradient(img1)
    #   2. Calculate temporal gradient
    I_t = np.subtract(img1, img2)
    
    height, width = img2.shape
    half_window_size = win_size // 2
    window_pixels_num = win_size ** 2
    u_v_list, y_x_list = [], []

    start = int(max(step_size, win_size) / 2)
    max_iterations_rows = height - int(max(step_size, win_size) / 2)
    max_iterations_cols = width - int(max(step_size, win_size) / 2)
    for i in range(start ,max_iterations_rows ,step_size):
        for j in range(start, max_iterations_cols, step_size):
            """
            For solving the:               # T   #   -1  # T  #
                                u         ###   ###     ###   ###
                                v   =   (#   # #   #)  #   #  ###
                to every pixel.
            """
            #   3. Extract local windows
            x_window = I_x[i - half_window_size: i + half_window_size + 1, j - half_window_size: j + half_window_size + 1]
            y_window = I_y[i - half_window_size: i + half_window_size + 1, j - half_window_size: j + half_window_size + 1]
            t_window = I_t[i - half_window_size: i + half_window_size + 1, j - half_window_size: j + half_window_size + 1]
            b = (-1) * t_window.reshape(window_pixels_num, 1)
            #   4. Compute A matrix of local windows
            A = np.hstack((x_window.reshape(window_pixels_num, 1), y_window.reshape(window_pixels_num, 1)))
            At_mult_A = np.dot(np.transpose(A), A)
            #   * Eigenvalues checking
            if not __eigenvalues_checking(At_mult_A):
                continue
            #   5. Solve the formula for the local windows to estimate optical flow (u, v)
            At_mult_b = np.dot(np.transpose(A), b)
            u_v = np.linalg.inv(At_mult_A) @ At_mult_b
            #   6. Update U(x, y) and V(x, y) with the estimated optical flow (u, v)
            y_x_list.append((j, i))
            u_v_list.append(u_v)
    return np.array(y_x_list).reshape(-1, 2), np.array(u_v_list).reshape(-1, 2)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    pass


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    pass


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    pass


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    pass


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    pass


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    pass


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    pass


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    pass


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    pass


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    pass

