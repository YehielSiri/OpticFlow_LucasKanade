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





from ex3_utils import *
import time


TRANSLATION_SMALL_STEP = np.array([[1, 0, -.2],
                                   [0, 1, -.1],
                                   [0, 0,   1]], dtype=np.float32)
TRANSLATION_BIG_STEP = np.array([[1, 0, -9],
                                 [0, 1, -7],
                                 [0, 0,  1]], dtype=np.float32)



# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def lkDemo(img_path):
    print("LK Demo")

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    img_2 = cv2.warpPerspective(img_1, TRANSLATION_SMALL_STEP, img_1.shape[::-1])

    # calc LK output
    st = time.time()
    pts, uv = opticalFlow(img_1.astype(np.float32), img_2.astype(np.float32), step_size=20, win_size=5)
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print(np.median(uv,0))
    print(np.mean(uv,0))

    displayOpticalFlow(img_2, pts, uv)


def hierarchicalkDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Hierarchical LK Demo")
    # For two images as input:
    # img_path1 = 'input/'
    # img_path2 = 'input/'
    # img_1 = cv2.cvtColor(cv2.imread(img_path1), cv2.COLOR_BGR2GRAY)
    # img_2 = cv2.cvtColor(cv2.imread(img_path2), cv2.COLOR_BGR2GRAY)
    
    # For one single image as input (making translation)
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    img_2 = cv2.warpPerspective(img_1, TRANSLATION_BIG_STEP, img_1.shape[::-1])

    st = time.time()
    STEP_SIZE, WIN_SIZE = 20, 9

    # calc hierarchical LK output
    UV = opticalFlowPyrLK(img_1.astype(float), img_2.astype(float), 6, stepSize=STEP_SIZE, winSize=WIN_SIZE)
    U, V = np.array(UV[:, :, 0]), np.array(UV[:, :, 1])
    pts, uv = Listed_u_v(U, V, stepSize=STEP_SIZE, winSize=WIN_SIZE)

    # Print results
    et = time.time()
    print("Time: {:.4f}".format(et - st))

    # Plot hierarchical LK Output
    displayOpticalFlow(img_2, pts, uv)


def compareLK(img_path):
    """
    ADD TEST
    Compare the two results from both functions.
    :param img_path: Image input
    :return:
    """
    print("Compare LK & Hierarchical LK")
    print("Compare LK Demo")
    # For two images as input:
    # img_path1 = 'input/'
    # img_path2 = 'input/'
    # img_1 = cv2.cvtColor(cv2.imread(img_path1), cv2.COLOR_BGR2GRAY)
    # img_2 = cv2.cvtColor(cv2.imread(img_path2), cv2.COLOR_BGR2GRAY)
    
    # For one single image as input (making translation)
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    img_2 = cv2.warpPerspective(img_1, TRANSLATION_SMALL_STEP, img_1.shape[::-1])

    STEP_SIZE, WIN_SIZE = 20, 9
    # Calc LK output
    pts, uv = opticalFlow(img_1.astype(float), img_2.astype(float), step_size=STEP_SIZE, win_size=WIN_SIZE)
    # calc hierarchical LK output
    UV = opticalFlowPyrLK(img_1.astype(float), img_2.astype(float), 6, stepSize=STEP_SIZE, winSize=WIN_SIZE)
    U, V = np.array(UV[:, :, 0]), np.array(UV[:, :, 1])
    ptsi, uvi = Listed_u_v(U, V, stepSize=STEP_SIZE, winSize=WIN_SIZE)

    # Print results
    # Plot both LK and hierarchical LK Output
    f, ax = plt.subplots(1, 2)
    ax[0].set_title('Optical Flow')
    ax[1].set_title('Iterative Optical Flow')
    ax[0].imshow(img_2, cmap='gray')
    ax[0].quiver(pts[:, 0], pts[:, 1], uv[:, 0], uv[:, 1], color='r')
    ax[1].imshow(img_2, cmap='gray')
    ax[1].quiver(ptsi[:, 0], ptsi[:, 1], uvi[:, 0], uvi[:, 1], color='r')
    plt.show()


def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')

    plt.show()


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def imageWarpingDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Image Warping Demo")

    pass


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def pyrGaussianDemo(img_path):
    print("Gaussian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 4
    gau_pyr = gaussianPyr(img, lvls)

    h, w = gau_pyr[0].shape[:2]
    canv_h = h
    widths = np.cumsum([w // (2 ** i) for i in range(lvls)])
    widths = np.hstack([0, widths])
    canv_w = widths[-1]
    canvas = np.zeros((canv_h, canv_w, 3))

    for lv_idx in range(lvls):
        h = gau_pyr[lv_idx].shape[0]
        canvas[:h, widths[lv_idx]:widths[lv_idx + 1], :] = gau_pyr[lv_idx]

    plt.imshow(canvas)
    plt.show()


def pyrLaplacianDemo(img_path):
    print("Laplacian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 7

    lap_pyr = laplaceianReduce(img, lvls)
    re_lap = laplaceianExpand(lap_pyr)

    f, ax = plt.subplots(2, lvls + 1)
    plt.gray()
    for i in range(lvls):
        ax[0, i].imshow(lap_pyr[i])
        ax[1, i].hist(lap_pyr[i].ravel(), 256, [lap_pyr[i].min(), lap_pyr[i].max()])

    ax[0, -1].set_title('Original Image')
    ax[0, -1].imshow(re_lap)
    ax[1, -1].hist(re_lap.ravel(), 256, [0, 1])
    plt.show()


def blendDemo():
    im1 = cv2.cvtColor(cv2.imread('input/sunset.jpg'), cv2.COLOR_BGR2RGB) / 255
    im2 = cv2.cvtColor(cv2.imread('input/cat.jpg'), cv2.COLOR_BGR2RGB) / 255
    mask = cv2.cvtColor(cv2.imread('input/mask_cat.jpg'), cv2.COLOR_BGR2RGB) / 255

    n_blend, im_blend = pyrBlend(im1, im2, mask, 4)

    f, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[0, 2].imshow(mask)
    ax[1, 0].imshow(n_blend)
    ax[1, 1].imshow(np.abs(n_blend - im_blend))
    ax[1, 2].imshow(im_blend)

    plt.show()

    cv2.imwrite('sunset_cat.png', cv2.cvtColor((im_blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    print("ID:", myID())

    img_path = 'input/boxMan.jpg'
    lkDemo(img_path)
    hierarchicalkDemo(img_path)
    compareLK(img_path)

    # imageWarpingDemo(img_path)

    # pyrGaussianDemo('input/pyr_bit.jpg')
    # pyrLaplacianDemo('input/pyr_bit.jpg')
    # blendDemo()


if __name__ == '__main__':
    main()
