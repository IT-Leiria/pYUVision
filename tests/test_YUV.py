import matplotlib.pyplot as plt
from pyuvision.pYUVision import YUV

def test_to_rgb():
    yuv = YUV('tests/lenna.yuv', (512, 512), 8)
    yuv.read()
    yuv.to_rgb()
    rgb = yuv.rgb
    assert rgb.shape == (512, 512, 3)
    plt.imshow(yuv.rgb)
    plt.show()


def show_yuv():
    yuv = YUV('tests/lenna.yuv', (512, 512), 8)
    yuv.read()
    assert yuv.shape == (512, 512, 3)
    fig = plt.figure(figsize=(10, 7)) 
    rows = 2
    columns = 2
    fig.add_subplot(rows, columns, 1) 
    plt.imshow(yuv.y, cmap='gray')
    fig.add_subplot(rows, columns, 2) 
    plt.imshow(yuv.u, cmap='Reds')
    fig.add_subplot(rows, columns, 3) 
    plt.imshow(yuv.v, cmap='Blues')
    fig.add_subplot(rows, columns, 4) 
    plt.imshow(yuv.y, cmap='gray')