import torch
import torch.nn as nn
from torchgan.models import DCGANGenerator
from torch.optim import Adam

img_size = 256  # 输出图像尺寸

class Generator(DCGANGenerator):
    def __init__(self):
        super(Generator, self).__init__(
            encoding_dims=100,#编码维度
            step_channels=40,#步长
            out_channels=3,#输出通道数
            out_size=img_size,#尺寸
            #noise_dim =50,#噪声张量
            nonlinearity=nn.LeakyReLU(0.3),
            last_nonlinearity=nn.Tanh()#最后一层的激活函数，使用了Tanh，将输出值缩放到[-1, 1]范围内。
        )

generator = Generator()
optimizer_gen = Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))#adam优化器
