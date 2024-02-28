#!/usr/bin/env python3
import numpy as np
from pyuvision.frame import Frame


class YUV:
    def __init__(self, filename, size, num_bit):
        self.width, self.height = size
        self.frame_len_luma = int((self.width * self.height * np.ceil(num_bit / 8)))
        self.frame_len_chroma = int(
            (self.width * self.height * np.ceil(num_bit / 8)) // 4
        )
        self.f = open(filename, "rb")
        self.shape_luma = (int(self.width), int(self.height))
        self.shape_chroma = (int(self.width / 2), int(self.height / 2))
        self.num_bits = num_bit

    def read(self, normalize=False):
        if self.num_bits == 8:
            dtype = np.uint8
        elif self.num_bits == 10 or self.num_bits == 16:
            dtype = np.uint16

        raw = self.f.read(self.frame_len_luma)
        y = np.frombuffer(raw, dtype=dtype).reshape(self.shape_luma[::-1])
        raw = self.f.read(self.frame_len_chroma)
        u = np.frombuffer(raw, dtype=dtype).reshape(self.shape_chroma[::-1])
        raw = self.f.read(self.frame_len_chroma)
        v = np.frombuffer(raw, dtype=dtype).reshape(self.shape_chroma[::-1])

        yuv = Frame((y, u, v), self.num_bits)

        return yuv
