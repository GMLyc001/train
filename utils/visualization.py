import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

def show_images(batch):
    plt.figure(figsize=(12, 12))#创建一个大小为 12x12 英寸的新图表
    plt.axis("off")# 关闭坐标轴，使图像更加清晰
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(batch, padding=2, normalize=True).cpu(), (1, 2, 0)))
    #创建一个网格布局的图像,间距为2，normalize=True 将图像数据标准化到 [0, 1] 区间
    plt.show()
