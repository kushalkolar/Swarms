import numpy as np
import pandas as pd
import trackpy
import cv2
import numba
from functools import partial
from multiprocessing import Pool


def locate(img: np.ndarray, diameter: int, minmass: int, maxmass: int, maxsize: float) -> pd.DataFrame:
    df = trackpy.locate(img, diameter=diameter, minmass=minmass, maxsize=maxsize)
    df = df[df['mass'] < maxmass]

    return df


def _locate_batch_wrapper(diameter: int, minmass: int, maxmass: int, maxsize: float, img_seq: np.ndarray) -> pd.DataFrame:
    df = trackpy.batch(img_seq, diameter=diameter, minmass=minmass, maxsize=maxsize)
    df = df[df['mass'] < maxmass]
    return df


def locate_batch(video: np.ndarray, diameter: int, minmass: int, maxmass: int, maxsize: float, threads: int) -> pd.DataFrame:
    pool = Pool(threads)
    dfs_list = pool.map(partial(_locate_batch_wrapper, diameter, minmass, maxmass, maxsize), np.array_split(video, threads, axis=0))
    pool.close()

    # Adjust the frame indices in the DataFrames from the trackpy.py in the wrapper func so that frames correspond to the orignal video
    frame_increment = 0
    for i in range(1, len(dfs_list)):
        frame_increment += dfs_list[i - 1][1][0]
        dfs_list[i][0]['frame'] += frame_increment

    df = pd.concat([t[0] for t in dfs_list])

    df = df[df['mass'] < maxmass]
    return df


def link(tracks: pd.DataFrame, filter_stubs: int = 0, **kwargs) -> pd.DataFrame:
    df = trackpy.link_df(tracks, **kwargs)

    if filter_stubs != 0:
        f = trackpy.filter_stubs(df, filter_stubs)
    else:
        f = df

    return f


def annotate_video(video: np.ndarray, frame_number: int, tracks: pd.DataFrame) -> np.ndarray:
    img = cv2.cvtColor(video[frame_number, :, :], cv2.COLOR_GRAY2RGB)

    for ix, row in tracks[tracks == frame_number].iterrows():
        x = int(row['x'])
        y = int(row['y'])
        cv2.circle(img, (x, y), 7, (255, 0, 0), 2)

    return img


def annotate_frame(img: np.ndarray, particles: pd.DataFrame) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for ix, row in particles.iterrows():
        x = int(row['x'])
        y = int(row['y'])
        cv2.circle(img, (x, y), 7, (255, 0, 0), 2)

    return img
