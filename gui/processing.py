from core.utils import *
from core import tracker
from .parameters import Parameters
import numpy as np


def process_frame(frame: np.ndarray, params: Parameters, mask=None) -> np.ndarray:
    if params.adjust_gamma:
        frame = adjust_gamma(frame, gamma=params.gamma)

    if params.use_clahe:
        clahe = get_clahe(clipLimit=params.clahe_clip_limit, tileGridSize=params.clahe_grid_size)
        frame = apply_clahe(clahe, frame)
    if mask is not None:
        frame = mask_arena(frame, mask)

    particles = tracker.locate(frame, diameter=params.diameter, minmass=params.minmass, maxmass=params.maxmass, maxsize=params.maxsize)

    annotated = tracker.annotate_frame(frame, particles)

    return annotated
