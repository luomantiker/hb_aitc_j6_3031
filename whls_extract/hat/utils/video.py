# Copyright (c) Horizon Robotics. All rights reserved.

from typing import List, Sequence

import cv2
import numpy as np
import tqdm


def encoding_frame_to_video(
    frame_list: List[np.ndarray],
    output_path: str,
    fps: int,
    size: Sequence[int] = None,
    encode_format: str = "MJPG",
):
    """
    Encode a list of frame (save as numpy.ndarray) to video.

    Args:
        frame_list: List of frame for encoding.
        output_path: Encoding video save path.
        fps: Fps of video.
        size: Video resolution, tuple of (width, height)
        encode_format: The format of encoding, refer to:
            https://learn.microsoft.com/en-us/windows/win32/medfound/video-fourccs
    """
    assert len(frame_list) > 0
    if size is None:
        height, width = frame_list[0].shape[:2]
        size = (width, height)
    src = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*encode_format), fps, size
    )
    for frame in tqdm.tqdm(frame_list, desc="Encoding..."):
        src.write(frame)
    src.release()
