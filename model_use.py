import torch
from generator import Generator
import torchvision.utils as vutils
import matplotlib.pyplot as plt

#模型文件路径
generator_model_path = r'D:\Anaconda\envs\test\torchgan_model\generator_epoch_0.pth'

# 创建生成器实例
generator = Generator()

# 加载训练好的生成器模型
generator.load_state_dict(torch.load(generator_model_path))

# 设置为评估模式
generator.eval()

# 噪声向量的维度应该与训练过程中使用的相同
noise_dim = 100  # 这里假设噪声维度为 100
num_images = 16  # 要生成的图像数量
noise = torch.randn(num_images, noise_dim, 1, 1)  # 创建噪声向量

# 生成图像
with torch.no_grad():
    fake_images = generator(noise)

# 可视化生成的图像
# 使用 torchvision.utils.make_grid 创建一个网格布局的图像
grid = vutils.make_grid(fake_images, padding=2, normalize=True)
plt.figure(figsize=(15, 15))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())  # 转换通道顺序以适合matplotlib
plt.show()
