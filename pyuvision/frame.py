#!/usr/bin/env python3

import copy
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


class Frame:
    def __init__(self, channels: tuple(), bit_depth: int):
        self.ch1 = None
        self.ch2 = None
        self.ch3 = None

        self.dtype = None
        self.bdp = bit_depth
        self.cformat = 400

        for i in range(len(channels)):
            c = channels[i]
            if isinstance(c, np.ndarray):
                if c.dtype == np.uint8:
                    if self.dtype is None:
                        self.dtype = np.uint8
                elif c.dtype == np.uint16:
                    if self.dtype is None:
                        self.dtype = np.uint16
                else:
                    raise ValueError(
                        "Invalid data type, must either have DType uint8 or uint16"
                    )
            else:
                raise ValueError("Invalid data type, must be numpy array")

            if i == 0:
                self.ch1 = copy.deepcopy(c.astype(self.dtype))
            if i == 1:
                self.ch2 = copy.deepcopy(c.astype(self.dtype))
                # If it has a 2nd channel, is not 400. Make value temporarly
                # different from 400 to infer the format below
                self.cformat = 4  # ...
            if i == 2:
                self.ch3 = copy.deepcopy(c.astype(self.dtype))

        # Infer subsampling
        # If it is different from 400, it means a second channel was found
        if self.cformat != 400:
            # NOTE: The following code to classify the chroma subsamplig was
            # generated using chatGPT, might have errors

            # Get the dimensions of the channels
            y_height, y_width = self.ch1.shape
            u_height, u_width = self.ch2.shape
            v_height, v_width = self.ch3.shape

            # Check if U and V channels have the same dimensions
            if (u_height, u_width) != (v_height, v_width):
                raise ValueError("Inconsistent chroma channel sizes")

            # Classify based on the ratios
            if (u_width, u_height) == (y_width, y_height):
                self.cformat = 444
            elif (u_width * 2, u_height) == (y_width, y_height):
                self.cformat = 422
            elif (u_width * 2, u_height * 2) == (y_width, y_height):
                self.cformat = 420
            elif (u_width * 4, u_height) == (y_width, y_height):
                self.cformat = 411
            elif (u_width, u_height * 2) == (y_width, y_height):
                self.cformat = 440
            else:
                raise ValueError("Unknown subsampling")

            self.orig_cformat = self.cformat

    def __getitem__(self, idx):
        if idx is not None:
            if self.cformat == 400:
                if len(idx) != 2:
                    raise IndexError(
                        "Invalid index: Gray image only, 2D index expected"
                    )
                return self.ch1[idx]
            else:
                if len(idx) != 3:
                    raise IndexError("Invalid index: Invalid index, 3D index expected")
                if idx[2] == 0:
                    return self.ch1[idx[:2]]
                if idx[2] == 1:
                    return self.ch2[idx[:2]]
                if idx[2] == 2:
                    return self.ch3[idx[:2]]
        else:
            raise IndexError("Frame not intialized")

    def normalize(self):
        if self.dtype == np.uint8:
            self.ch1 = self.ch1.astype(np.float32) / 255
            self.ch2 = self.ch2.astype(np.float32) / 255
            self.ch3 = self.ch3.astype(np.float32) / 255
        else:
            self.ch1 = self.ch1.astype(np.float32) / (2**self.bdp - 1)
            self.ch2 = self.ch2.astype(np.float32) / (2**self.bdp - 1)
            self.ch3 = self.ch3.astype(np.float32) / (2**self.bdp - 1)
        self.dtype = np.float32

    def chroma_upsample(self):
        if self.cformat == 400:
            raise ValueError("Gray image only, chroma upsampling not possible")
        elif self.ch1.shape == self.ch2.shape:
            raise ValueError("Image is already 444")
        else:
            h, w = self.ch1.shape

            self.ch2 = np.array(
                Image.fromarray(self.ch2).resize((w, h), Image.Resampling.LANCZOS)
            )
            self.ch3 = np.array(
                Image.fromarray(self.ch3).resize((w, h), Image.Resampling.LANCZOS)
            )
            self.cformat = 444

    def to_rgb(self):
        if self.cformat != 400:
            if self.cformat == 444:
                dtype = self.dtype

                if dtype != np.float32:
                    self.normalize()

                y = copy.deepcopy(self.ch1).astype(np.float32)
                u = copy.deepcopy(self.ch2).astype(np.float32) - 0.5
                v = copy.deepcopy(self.ch3).astype(np.float32) - 0.5

                # Perform the conversion
                self.ch1 = (y + 1.402 * v).clip(0, 1)
                self.ch2 = (y - 0.344136 * u - 0.714136 * v).clip(0, 1)
                self.ch3 = (y + 1.772 * u).clip(0, 1)

                if dtype != np.float32:
                    self.ch1 = (self.ch1 * (2**self.bdp - 1)).astype(dtype)
                    self.ch2 = (self.ch2 * (2**self.bdp - 1)).astype(dtype)
                    self.ch3 = (self.ch3 * (2**self.bdp - 1)).astype(dtype)

                    self.dtype = dtype

            else:
                raise ValueError(
                    "Chroma format is not 444. Use chroma_upsample() first"
                )
        else:
            raise ValueError("Gray image only, conversion to RGB not possible")

    def to_numpy(self):
        if self.cformat == 400:
            return self.ch1
        else:
            if self.cformat == 444:
                return np.stack([self.ch1, self.ch2, self.ch3], axis=2)
            else:
                raise ValueError(
                    "Chroma format is not 444. Use chroma_upsample() first"
                )

    def plot(self):
        if self.cformat == 400:
            plt.imshow(self.ch1, cmap="gray")
        else:
            f = copy.deepcopy(self)
            f.chroma_upsample()
            f.to_rgb()
            plt.imshow(f.to_numpy())
        plt.show()
