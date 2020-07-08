import numpy as np
import cv2
import skvideo.io
from gui.parameters import Parameters
import os
from tqdm import tqdm


def load_video(path: str, **kwargs) -> np.ndarray:
    """

    :param path:
    :param kwargs: passed to skvideo.io.vread
    :return:
    """
    return skvideo.io.vread(path, as_grey=True, **kwargs)[:, :, :, 0]


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


def write_masked_video(params: Parameters):

    # get the first frame to create the mask
    cap = cv2.VideoCapture(params.video_path)

    for i in range(100):
        cap.read()

    r, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # create a mask
    mask = get_mask(
        gray,
        param1=params.circle_param1,
        param2=params.circle_param2,
        min_radius=params.circle_minradius,
        max_radius=params.circle_maxradius
    )

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(
        os.path.dirname(params.video_path),
        'circle_masked',
        f'circle_masked-{os.path.basename(params.video_path)}'
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=False)

    cap = cv2.VideoCapture(params.video_path)

    print("Writing frames...")
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    pbar = tqdm(total=n_frames)

    while cap.isOpened():
        r, img = cap.read()

        if r:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            out_img = mask_arena(gray, mask)
            out.write(out_img)
            pbar.update(1)

        else:
            break

    cap.release()
    out.release()
