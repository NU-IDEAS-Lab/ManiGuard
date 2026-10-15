"""Bounded-memory RGB video encoding, independent of simulation and recording."""

from fractions import Fraction
from pathlib import Path

import numpy as np


class StreamingVideoWriter:
    """Encode each RGB frame immediately; close flushes the bounded codec buffer."""

    def __init__(self, path, *, fps, frame_hw):
        import av

        self._av = av
        self._shape = (*frame_hw, 3)
        self._closed = False
        if fps <= 0 or any(size <= 0 or size % 2 for size in frame_hw):
            raise ValueError("Video needs positive FPS and even, positive frame dimensions")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(path), "w")
        try:
            self._stream = self._container.add_stream("libx264", rate=Fraction(str(fps)))
            self._stream.height, self._stream.width = frame_hw
            self._stream.pix_fmt = "yuv420p"
            self._stream.options = {"crf": "14"}
        except BaseException:
            self._container.close()
            raise

    def append(self, rgb):
        if self._closed:
            raise ValueError("Video writer is closed")
        if rgb.shape != self._shape or rgb.dtype != np.uint8:
            raise ValueError(f"Expected uint8 RGB shape {self._shape}, got {rgb.shape}/{rgb.dtype}")
        for packet in self._stream.encode(self._av.VideoFrame.from_ndarray(rgb, format="rgb24")):
            self._container.mux(packet)

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                for packet in self._stream.encode():
                    self._container.mux(packet)
            finally:
                self._container.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
