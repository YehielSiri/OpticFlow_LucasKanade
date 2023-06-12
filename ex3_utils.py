"""
        '########:'##::::'##::'########:::
         ##.....::. ##::'##:::::.....##:::
         ##::::::::. ##'##:::::::::::##:::
         ######:::::. ###::::. ########:::
         ##...:::::: ## ##:::::......##:::
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

# ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^   Auxolarity functions   ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^
def __gradient(img: np.ndarray):
    I_x = cv2.filter2D(img, -1, GRADIENT)
    I_y = cv2.filter2D(img, -1, GRADIENT.T)
    return I_x, I_y

def is_window_valid(eigen_value: np.ndarray) -> bool:
    """
    Checking if the eigenvalues is valid.
    """
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

def __blur_image(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    # Kernel size must be odd and positive
    if ((k_size % 2) == 0 or k_size < 0):
        print ("The kernel dimension must be odd and positive")
        return in_image
    
    # Build a gaussian kernel. cv2 create an 1D one!
    kernel1D = cv2.getGaussianKernel(k_size, -1)
    kernel2D = kernel1D @ kernel1D.transpose()

    return cv2.filter2D(in_image, -1, kernel2D, borderType = cv2.BORDER_REPLICATE)

def __gauss_expand(img: np.ndarray, target_shape: np.shape):
    height, width = img.shape[0] * 2, img.shape[1] * 2
    gaus_expended_prev_level = np.zeros(target_shape)
    for i in range(0, width, 2):
        for j in range(0, height, 2):
            gaus_expended_prev_level[j, i] = img[j // 2][i // 2]
    gaus_expended_prev_level = np.clip(__blur_image(gaus_expended_prev_level, 5) * 4, 0, 1)
    return gaus_expended_prev_level

def __optical_Flow(im1: np.ndarray, im2: np.ndarray, step_size: int,
                win_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    An auxilarity function to both, opticalFlow() & opticalFlowPyrLK()
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points, [[dU]...] for each points, [[dV]...] for each points
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
    u = np.zeros(img1.shape)
    v = np.zeros(img1.shape)


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
            u[i - half_window_size: i + half_window_size + 1, j - half_window_size: j + half_window_size + 1] += u_v[0]
            v[i - half_window_size: i + half_window_size + 1, j - half_window_size: j + half_window_size + 1] += u_v[1]
    return np.array(y_x_list).reshape(-1, 2), np.array(u_v_list).reshape(-1, 2), u, v

# ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^







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
    img1, img2 = im1, im2
    u, v = np.zeros(img1.shape), np.zeros(img1.shape)

    y_x_list, u_v_list, u, v = __optical_Flow(img1.astype(np.float32), img2.astype(np.float32), step_size, win_size)
    return y_x_list, u_v_list



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
    gaus_pyr1 = gaussianPyr(img1, k)
    gaus_pyr2 = gaussianPyr(img2, k)
    u = np.zeros(gaus_pyr1[k - 1].shape)
    v = np.zeros(gaus_pyr1[k - 1].shape)
    for i in range(k - 1, -1, -1):
        # warp image 1 towards image 2 using u, v (first iteration insignificant since u = v = 0)
        warped = warp_u_v(gaus_pyr1[i], u, v)
        # calc winSize, stepSize for the current lvl
        win_size = int(winSize / (i + 1)) if int(winSize / (i + 1)) % 2 == 1 else int(winSize / (i + 1)) + 1
        step_size = int(stepSize / (i + 1)) if int(stepSize / (i + 1)) % 2 == 1 else int(stepSize / (i + 1)) + 1
        # calc LK for the current lvl
        d_u, d_v = np.array(__optical_Flow_u_v(warped, gaus_pyr2[i], step_size=step_size, win_size=int(win_size)))
        # blur to accurate result
        d_u = __blur_image(d_u * 4, 5)
        d_v = __blur_image(d_v * 4, 5)
        # avoid errors
        d_u = np.where(abs(d_u - 0) < 5, d_u, 0)
        d_v = np.where(abs(d_v - 0) < 5, d_v, 0)
        # merge current d_u, d_v with total u, v
        d_u, v = merge_u_v(u, v, d_u * 2, d_v * 2)
        # expand u, v dimension to the next level (except last iteration [i=0] which it the original shape)
        if i != 0:
            u = __gauss_expand(u, gaus_pyr1[i - 1].shape) * 4
            v = __gauss_expand(v, gaus_pyr1[i - 1].shape) * 4
    u = np.round(to_shape(u, gaus_pyr1[0].shape))
    v = np.round(to_shape(v, gaus_pyr1[0].shape))
    # return d_u, d_v as pairs
    return np.stack((u, v), axis=2)


# ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^   Auxolarity functions   ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^
def __optical_Flow_u_v(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> tuple[np.ndarray, np.ndarray]:
    """
        An auxilarity function of 'opticalFlowPyrLK()' which return only the u & v for each pixel.
    """
    img1, img2 = im1, im2
    u, v = np.zeros(img1.shape), np.zeros(img1.shape)

    y_x_list, u_v_list, u, v = __optical_Flow(img1.astype(np.float32), img2.astype(np.float32), step_size, win_size)
    return u, v

def Listed_u_v(d_u: np.ndarray, d_v: np.ndarray, stepSize: int, winSize: int):
    u_v_list, y_x_list = [], []
    height, width = d_u.shape
    win_iterator = int(max(stepSize, winSize) / 2)
    for i in range(win_iterator, height - win_iterator, stepSize):
        for j in range(win_iterator, width - win_iterator, stepSize):
            if 0 <= j + d_v[i][j] < width and 0 <= i + d_u[i][j] < height \
                    and (int(d_v[i][j]) != 0 or int(d_u[i][j]) != 0):
                y_x_list.append((j, i))
                u_v_list.append((int(d_u[i][j]), int(d_v[i][j])))
    return np.array(y_x_list).reshape(-1, 2), np.array(u_v_list).reshape(-1, 2)

def merge_u_v(u: np.ndarray, v: np.ndarray, d_u: np.ndarray, d_v: np.ndarray):
    for y in range(u.shape[1]):
        for x in range(u.shape[0]):
            move_u, move_v = int(u[x][y]), int(v[x][y])
            if 0 <= y + move_v < d_u.shape[1] and 0 <= x + move_u < d_u.shape[0] \
                    and 0 <= y < u.shape[1] and 0 <= x < u.shape[0]:
                u[x][y] += d_u[x + move_u][y + move_v]
                d_u[x + move_u][y + move_v] = 0
                v[x][y] += d_v[x + move_u][y + move_v]
                d_v[x + move_u][y + move_v] = 0
    return u, v

def warp_u_v(image: np.ndarray, u: np.ndarray, v: np.ndarray):
    result = np.zeros(image.shape) - 1
    for y in range(image.shape[1] - 2):
        for x in range(image.shape[0] - 2):
            if x < u.shape[0] and y < v.shape[1] and int(x + u[x][y]) < result.shape[0] and int(y + v[x][y]) < \
                    result.shape[1] and int(x + u[x][y]) >= 0 and int(y + v[x][y]) >= 0:
                result[int(x + u[x][y])][int(y + v[x][y])] = image[x][y]
    mask = np.array([[1 if result[j][i] == -1 else 0 for i in range(image.shape[1])] for j in range(image.shape[0])])
    interpolated = cv2.inpaint(cv2.convertScaleAbs(result), cv2.convertScaleAbs(mask), 1, cv2.INPAINT_TELEA)
    return interpolated

def to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = max((y_ - y), 0)
    x_pad = max((x_ - x), 0)
    return np.pad(a, ((y_pad // 2, y_pad // 2 + y_pad % 2),
                      (x_pad // 2, x_pad // 2 + x_pad % 2)), mode='constant')

# ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------

# ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^   Auxolarity functions   ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^
def maxCorrelationPoint(im1: np.ndarray, im2: np.ndarray):
    """
    https://stackoverflow.com/questions/58181398/how-to-find-correlation-between-two-images-using-numpy
    """
    # Get rid of the averages, otherwise the results are not good
    rows, cols = max(im1.shape[0], im2.shape[0]), max(im1.shape[1], im2.shape[1])
    im1_gray = im1 - np.mean(im1)
    im2_gray = im2 - np.mean(im2)
    im1_gray = np.pad(im1_gray, [(0, max(0, rows - im1.shape[0])),
                                 (0, max(0, cols - im1.shape[1]))], mode='constant')
    im2_gray = np.pad(im2_gray, [(0, max(0, rows - im2.shape[0])),
                                 (0, max(0, cols - im2.shape[1]))], mode='constant')
    # Calculate the correlation image (without scipy)
    pad = np.max(im1_gray.shape) // 2
    fft1 = np.fft.fft2(np.pad(im1_gray, pad))
    fft2 = np.fft.fft2(np.pad(im2_gray, pad))
    prod = fft1 * fft2.conj()
    result_full = np.fft.fftshift(np.fft.ifft2(prod))
    corr = result_full.real[1 + pad:-pad + 1, 1 + pad:-pad + 1]
    max_point_value = np.argmax(corr)
    y, x = np.unravel_index(max_point_value, corr.shape)
    return y, x, np.max(corr)

def findRotation(im1: np.ndarray, im2: np.ndarray):
    rows, cols = im1.shape
    # Find the angle with the highest correlation point
    max_error, theta_res = 0, 0
    for i in range(0, 360):
        theta = i * 0.0174532925  # degree to radian
        rotate_matrix = np.float32([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0]])
        rotated = cv2.warpAffine(im1, rotate_matrix, (rows, cols))
        y, x, error = maxCorrelationPoint(rotated, im2)
        if error > max_error:
            max_error = error
            theta_res = theta
    return theta_res

# ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    # Calc U, V Matrices using LK iterative (pyramids)
    step_size, win_size = 20, 9
    UV = opticalFlowPyrLK(im1.astype(float), im2.astype(float), 6, stepSize=step_size, winSize=win_size)
    U, V = np.array(UV[:, :, 0]), np.array(UV[:, :, 1])
    pts, uv = Listed_u_v(U, V, stepSize=step_size, winSize=win_size)
    rows, cols = im1.shape
    # Find specific u, v that's giving the minimum error
    min_error = np.square(np.abs(im2 - im1)).mean()
    res_u, res_v = 0, 0
    for u, v in uv:
        cv2_warp_matrix = np.float32([[1, 0, u], [0, 1, v]])
        cv2_warping_res = cv2.warpAffine(im1, cv2_warp_matrix, (cols, rows))
        error = np.square(np.abs(im2 - cv2_warping_res)).mean()
        if error < min_error:
            min_error = error
            res_u, res_v = u, v
    # Result as warping matrix
    warping_mat = np.array([[1, 0, res_u], [0, 1, res_v], [0, 0, 1]])
    return warping_mat


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    rows, cols = im1.shape
    theta = findRotation(im1, im2)
    rotate_matrix = np.float32([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0]])
    rotated = cv2.warpAffine(im1, rotate_matrix, (cols, rows))
    # Calc U, V using LK iterative (pyramids)
    translation_mat = findTranslationLK(rotated, im2)
    u, v = translation_mat[0, 2], translation_mat[1, 2]
    # Result as warping matrix
    warping_mat = np.array([[np.cos(theta), -np.sin(theta), u],
                            [np.sin(theta), np.cos(theta), v],
                            [0, 0, 1]])
    return warping_mat


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    I have learned from:
    https://stackoverflow.com/questions/58174390/how-to-detect-image-translation-with-only-numpy-and-pil
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    y, x, max_point_value = maxCorrelationPoint(im1, im2)
    # Calculate maximum image size for im1 and im2
    rows, cols = max(im1.shape[0], im2.shape[0]), max(im1.shape[1], im2.shape[1])
    # Calculate the distance from the maximum error point to the midpoint
    y_distance = rows // 2 - y
    x_distance = cols // 2 - x
    # Result as warping matrix
    warping_mat = np.array([[1, 0, x_distance], [0, 1, y_distance], [0, 0, 1]])
    return warping_mat


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    I have learned from:
    https://stackoverflow.com/questions/23619269/calculating-translation-value-and-rotation-angle-of-a-rotated-2d-image
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    rows, cols = im1.shape
    theta = findRotation(im1, im2)
    rotate_matrix = np.float32([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0]])
    rotated = cv2.warpAffine(im1, rotate_matrix, (cols, rows))
    # Calc U, V using Corr
    translation_mat = findTranslationCorr(rotated, im2)
    u, v = translation_mat[0, 2], translation_mat[1, 2]
    # Result as warping matrix
    warping_mat = np.array([[np.cos(theta), -np.sin(theta), u],
                            [np.sin(theta), np.cos(theta), v],
                            [0, 0, 1]])
    return warping_mat


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    result = np.zeros(im1.shape)
    for i in range(0, result.shape[0]):
        for j in range(0, result.shape[1]):
            """
            Usually we use inverse matrix of T which is T^1 (using np.linalg.inv(T))
            That, in condition of the transform which has an inverse!
            But this time we want to perform warping backwards so we use (T^1)^1 which is T
            [since p2=(T^1)*p1 we get T*p2=T*(T^1)*p1 = T*p2=p1]
            """
            new_coordinates = T.dot(np.array([j, i, 1]))
            new_j, new_i = int(new_coordinates[0]), int(new_coordinates[1])
            if 0 <= new_i < im2.shape[0] and 0 <= new_j < im2.shape[1]:
                result[i, j] = im2[new_i, new_j]
    return result


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
    pyr = []
    height, width = img.shape[0], img.shape[1]
    for level in range(levels):
        pyr.append(img)
        img = __blur_image(img, 5)
        img = np.array([[img[j][i] for i in range(0, width, 2)] for j in range(0, height, 2)])
        height, width = int(height / 2), int(width / 2)
    return pyr


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

