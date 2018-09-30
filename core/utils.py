import numpy as np
import cv2
import skvideo.io


def load_video(path: str) -> np.ndarray:
    return skvideo.io.vread(path, as_grey=True)[:, :, :, 0]


def adjust_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    :param img:     2D numpy array
    :param gamma:
    :return: 2D numpy array
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)


def get_clahe(clipLimit=2.0, tileGridSize=(8, 8)):
    return cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)


def apply_clahe(clahe, img: np.ndarray) -> np.ndarray:
    return clahe.apply(img)


def get_mask(img: np.ndarray, param1: float, param2: int, min_radius: int, max_radius: int) -> np.ndarray:
    """

    :param img:         2D numpy array
    :param param1:
    :param param2:
    :param min_radius:
    :param max_radius:
    :param zeros_array: 2D numpy array of zeros with same shape as img
    :return:            2D numpy array
    """

    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, param1, param2, minRadius=min_radius, maxRadius=max_radius)

    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(mask, (x, y), r, (255), -1)

    return mask


def mask_arena(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(img, mask)


def mask_video(video: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    :param video: 3D numpy array, first index of the shape is the video frame indices
    :param mask:  mask numpy array returned from get_mask()
    :return: 3D numpy array of masked video
    """

    for i in range(video.shape[0]):
        video[i, :, :] = cv2.bitwise_and(video[i, :, :], mask)

    return video