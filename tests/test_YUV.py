import matplotlib.pyplot as plt
import numpy as np
from pyuvision.yuv import YUV


def test_to_rgb():
    yuv = YUV("tests/yuv/lena.yuv", (512, 512), 8)
    f = yuv.read()
    f.chroma_upsample()
    f.to_rgb()
    f = f.to_numpy()
    assert isinstance(f, np.ndarray)
    assert f.shape == (512, 512, 3)


def test_show_yuv():
    yuv = YUV("tests/yuv/lena.yuv", (512, 512), 8)
    f = yuv.read()
    f.plot()
