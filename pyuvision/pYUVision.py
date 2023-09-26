import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image


class YUV:
    def __init__(self, filename, size, num_bit):
        self.width, self.height = size
        self.frame_len_luma = int(
            (self.width * self.height*np.ceil(num_bit/8)))
        self.frame_len_chroma = int(
            (self.width * self.height*np.ceil(num_bit/8))//4)
        self.f = open(filename, 'rb')
        self.shape_luma = (int(self.width), int(self.height))
        self.shape_chroma = (int(self.width/2), int(self.height/2))
        self.num_bits = num_bit

    def __normalise(self, I):
        # return (I.astype(float) - float(np.min(I)))/(float(np.max(I)))
        aux = I.astype(float) - float(np.min(I))
        aux = aux/np.max(aux)
        return aux

    def read(self):
        if self.num_bits == 8:
            dtipo = np.uint8
        elif self.num_bits == 10 or self.num_bits == 16:
            dtipo = np.uint16

        raw = self.f.read(self.frame_len_luma)
        y = np.frombuffer(raw, dtype=dtipo).reshape(self.shape_luma[::-1])
        raw = self.f.read(self.frame_len_chroma)
        u = np.frombuffer(raw, dtype=dtipo).reshape(self.shape_chroma[::-1])
        raw = self.f.read(self.frame_len_chroma)
        v = np.frombuffer(raw, dtype=dtipo).reshape(self.shape_chroma[::-1])

        u = np.array(Image.fromarray(u).resize(
            self.shape_luma, Image.Resampling.LANCZOS))
        v = np.array(Image.fromarray(v).resize(
            self.shape_luma, Image.Resampling.LANCZOS))

        if self.num_bits == 10 or self.num_bits == 16:
            y = y/(2 ^ self.num_bits)
            u = u/(2 ^ self.num_bits)
            v = v/(2 ^ self.num_bits)

        # self.yuv = np.uint8(np.stack([y, u, v], axis=2))
        self.yuv = self.__normalise(np.stack([y, u, v], axis=2))

    def to_rgb(self):
        r = self.yuv[:, :, 0] + 1.13983 * self.yuv[:, :, 2]
        g = self.yuv[:, :, 0] - 0.39465 * \
            self.yuv[:, :, 1] - 0.58060 * self.yuv[:, :, 2]
        b = self.yuv[:, :, 0] - 2.03211 * self.yuv[:, :, 1]

        # self.rgb = np.uint8(np.stack([r, g, b], axis=2))
        self.rgb = self.__normalise(np.stack([r, g, b], axis=2))

    def show_vvc_overlay(self, file_path, linewidth=0.5):
        fig, ax = plt.subplots()
        # TODO make RGB work
        # self.to_rgb()
        # ax.imshow(self.rgb)
        ax.imshow(self.yuv[:, :, 0], cmap='gray')

        if not isinstance(file_path, list):
            with open(file_path, 'r') as f:
                for line in f:
                    if not ('chroma' in line):
                        p = line.split()

                        x = float(p[0].strip()[2:-1])
                        y = float(p[1].strip()[2:-1])
                        w = float(p[2].strip()[2:-1])
                        h = float(p[3].strip()[2:-1])

                        r = patches.Rectangle(
                            (x, y), w, h, linewidth=0.5, edgecolor='r', facecolor='none')

                        ax.add_patch(r)
        else:
            colours = ['r', 'b', 'g']
            lwidths = [2, 1, 3]
            for i in range(len(file_path)):
                with open(file_path[i], 'r') as f:
                    for line in f:
                        if not ('chroma' in line):
                            p = line.split()

                            x = float(p[0].strip()[2:-1])
                            y = float(p[1].strip()[2:-1])
                            w = float(p[2].strip()[2:-1])
                            h = float(p[3].strip()[2:-1])

                            r = patches.Rectangle(
                                (x, y), w, h, linewidth=lwidths[i], edgecolor=colours[i], facecolor='none', alpha=0.5)

                            ax.add_patch(r)

        plt.show()
