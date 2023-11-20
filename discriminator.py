import torch
import torch.nn as nn
from torchgan.models import DCGANDiscriminator
from torch.optim import Adam

img_size = 256  # 输入图像尺寸

class Discriminator(DCGANDiscriminator):
    def __init__(self):
        super(Discriminator, self).__init__(
            in_channels=3,#输入通道数,通常为RGB图像，所以为3
            in_size=img_size,#输入图像尺寸
            step_channels=40,#步长通道数,影响网络的深度或复杂度
            nonlinearity=nn.LeakyReLU(0.3),#激活函数,使用了LeakyReLU，即带有负斜率的ReLU激活函数
            last_nonlinearity=nn.LeakyReLU(0.2)
        )

discriminator = Discriminator()
optimizer_disc = Adam(discriminator.parameters(), lr=0.0006, betas=(0.5, 0.999))
'''
创建了一个Adam优化器，用于优化判别器的参数。
传递了判别器的参数 (discriminator.parameters())，学习率为0.0006，betas参数设置为(0.5, 0.999)。
'''
